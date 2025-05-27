# src/phase_4_synthesizer.py

import logging
import os
import traceback

try:
    from moviepy.editor import (
        VideoFileClip,
        AudioFileClip,
        concatenate_videoclips,
        CompositeVideoClip, # Needed for placing resized clip on canvas
        ColorClip,
    )
    from moviepy.video.fx.all import resize
    MOVIEPY_AVAILABLE = True
except ImportError:
    print("--------------------------------------------------------------------")
    print("IMPORTANT: MoviePy library not found or an issue occurred importing it.")
    print("Please install it to enable video synthesis: pip install moviepy")
    print("Ensure FFmpeg is also installed and accessible in your system's PATH.")
    print("--------------------------------------------------------------------")
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None
    AudioFileClip = None
    concatenate_videoclips = None
    CompositeVideoClip = None
    ColorClip = None
    resize = None

logger = logging.getLogger(__name__)

class Synthesizer:
    """
    Synthesizes the final video based on an Edit Decision List (EDL) and narration audio.
    """
    def __init__(self, config):
        """
        Initializes the Synthesizer.

        Args:
            config (dict): The application configuration dictionary.
        """
        self.config = config
        # Corrected path key based on user's config.yaml
        self.video_clips_base_dir = self.config.get("paths", {}).get("input_video_dir") # Should be input_video_dir as per config

        # Video output settings from config's "settings" section
        settings_config = self.config.get("settings", {})
        self.raw_output_resolution_setting = settings_config.get("output_video_resolution", "1920x1080")
        self.output_fps = settings_config.get("output_video_fps", 25)

        # Other synthesizer parameters
        self.video_codec = settings_config.get("video_codec", "libx264")
        self.audio_codec = settings_config.get("audio_codec", "aac")
        self.video_preset = settings_config.get("video_preset", "medium")
        self.ffmpeg_threads = settings_config.get("ffmpeg_threads", 4)
        self.temp_audiofile_name = settings_config.get("temp_audiofile_name", "temp_synthesizer_audio.aac")

        self.output_width = None
        self.output_height = None


    def _determine_output_resolution(self, edl):
        """
        Determines the output resolution. If "auto", tries to get from the first EDL video.
        Sets self.output_width and self.output_height.
        """
        default_width, default_height = 1920, 1080

        if isinstance(self.raw_output_resolution_setting, str) and \
           self.raw_output_resolution_setting.lower() == "auto":
            logger.info("Output resolution set to 'auto'. Attempting to detect from first video clip in EDL.")
            first_clip_path = None
            if edl: # Ensure EDL is not empty
                for event in edl:
                    video_filename = event.get("matched_video_filename")
                    if video_filename and self.video_clips_base_dir: # Ensure base_dir is configured
                        # 创建文件夹映射缓存
                        folder_mapping = {}
                        try:
                            for folder_name in os.listdir(self.video_clips_base_dir):
                                folder_path = os.path.join(self.video_clips_base_dir, folder_name)
                                if os.path.isdir(folder_path):
                                    folder_mapping[folder_name] = folder_path
                        except Exception as e:
                            logger.error(f"Error scanning video clips directory: {e}")
                        
                        # 查找匹配的文件夹
                        for folder_name in folder_mapping:
                            if video_filename.startswith(folder_name):
                                # 文件应该在这个文件夹中
                                potential_path = os.path.join(folder_mapping[folder_name], video_filename)
                                if os.path.exists(potential_path):
                                    first_clip_path = potential_path
                                    break
                        
                        # 如果没找到，尝试直接路径
                        if not first_clip_path:
                            direct_path = os.path.join(self.video_clips_base_dir, video_filename)
                            if os.path.exists(direct_path):
                                first_clip_path = direct_path
                        
                        if first_clip_path:
                            break
            
            if first_clip_path:
                temp_clip = None
                try:
                    logger.info(f"Loading first clip for resolution detection: {first_clip_path}")
                    temp_clip = VideoFileClip(first_clip_path)
                    self.output_width = temp_clip.size[0]
                    self.output_height = temp_clip.size[1]
                    logger.info(f"Auto-detected resolution from '{os.path.basename(first_clip_path)}': {self.output_width}x{self.output_height}")
                except Exception as e:
                    logger.warning(f"Could not load or get size from first video clip '{first_clip_path}' for 'auto' resolution: {e}. Falling back to default.")
                    self.output_width = default_width
                    self.output_height = default_height
                finally:
                    if temp_clip:
                        temp_clip.close()
            else:
                logger.warning("No valid video clips found in EDL (or input_video_dir not set) to determine 'auto' resolution. Falling back to default.")
                self.output_width = default_width
                self.output_height = default_height
        else: 
            try:
                parts = str(self.raw_output_resolution_setting).lower().split('x')
                if len(parts) == 2:
                    self.output_width = int(parts[0])
                    self.output_height = int(parts[1])
                    logger.info(f"Using specified output resolution: {self.output_width}x{self.output_height}")
                else:
                    raise ValueError("Resolution string format is incorrect.")
            except ValueError as e:
                logger.warning(f"Invalid format for 'output_video_resolution': '{self.raw_output_resolution_setting}'. Expected 'WIDTHxHEIGHT' (e.g., '1920x1080') or 'auto'. Falling back to default. Error: {e}")
                self.output_width = default_width
                self.output_height = default_height
        
        if not self.output_width or not self.output_height:
             logger.error(f"Output resolution could not be determined. Falling back to {default_width}x{default_height}.")
             self.output_width = default_width
             self.output_height = default_height

    def _find_video_path(self, video_filename):
        if not video_filename or not self.video_clips_base_dir:
            return None
        
        # 尝试从文件夹映射中查找
        if not hasattr(self, '_folder_mapping_cache'):
            # 创建文件夹映射缓存
            self._folder_mapping_cache = {}
            try:
                for folder_name in os.listdir(self.video_clips_base_dir):
                    folder_path = os.path.join(self.video_clips_base_dir, folder_name)
                    if os.path.isdir(folder_path):
                        self._folder_mapping_cache[folder_name] = folder_path
            except Exception as e:
                logger.error(f"扫描视频目录时出错: {e}")
        
        # 查找匹配的文件夹
        for folder_name, folder_path in self._folder_mapping_cache.items():
            if video_filename.startswith(folder_name):
                # 可能在这个文件夹中
                potential_path = os.path.join(folder_path, video_filename)
                if os.path.exists(potential_path):
                    return potential_path
        
        # 尝试直接在基础目录中查找
        direct_path = os.path.join(self.video_clips_base_dir, video_filename)
        if os.path.exists(direct_path):
            return direct_path
        
        return None

    def synthesize_video(self, edl, narration_audio_path, output_video_path):
        """
        基于编辑决策列表(EDL)和旁白音频生成最终视频。
        
        Args:
            edl (list): 编辑决策列表，包含视频段信息。
            narration_audio_path (str): 旁白音频文件路径。
            output_video_path (str): 输出视频文件路径。
            
        Returns:
            bool: 成功返回True，失败返回False。
        """
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy库不可用。无法合成视频。")
            return False
        if not self.video_clips_base_dir:
            logger.error("未配置输入视频片段目录（'input_video_dir'）。无法找到B-roll素材。")
            return False
        if not os.path.exists(narration_audio_path):
            logger.error(f"找不到旁白音频文件: {narration_audio_path}")
            return False
        if not edl:
            logger.error("编辑决策列表（EDL）为空。没有内容可合成。")
            return False

        self._determine_output_resolution(edl)

        logger.info(f"开始视频合成。输出将保存到: {output_video_path}")
        logger.info(f"目标分辨率: {self.output_width}x{self.output_height}, FPS: {self.output_fps}")
        logger.info(f"使用旁白音频: {narration_audio_path}")

        # 直接检查视频文件夹结构并输出详细信息
        logger.info(f"视频文件基础目录: {self.video_clips_base_dir}")
        if os.path.exists(self.video_clips_base_dir):
            logger.info(f"视频基础目录存在。内容:")
            try:
                folder_names = os.listdir(self.video_clips_base_dir)
                logger.info(f"基础目录内容: {', '.join(folder_names)}")
                
                # 检查子文件夹
                for folder_name in folder_names:
                    full_folder_path = os.path.join(self.video_clips_base_dir, folder_name)
                    if os.path.isdir(full_folder_path):
                        files_in_folder = os.listdir(full_folder_path)
                        logger.info(f"子文件夹 '{folder_name}' 包含 {len(files_in_folder)} 个文件")
                        # 输出前5个文件名作为示例
                        if files_in_folder:
                            logger.info(f"示例文件: {', '.join(files_in_folder[:5])}")
            except Exception as e:
                logger.error(f"列出视频目录内容时出错: {e}")
        else:
            logger.error(f"视频基础目录不存在: {self.video_clips_base_dir}")

        all_moviepy_clips = [] # 跟踪所有视频片段以便适当关闭

        try:
            narration_clip = AudioFileClip(narration_audio_path)

            video_subclips_for_concatenation = []
            total_expected_video_duration = 0.0

            # 读取EDL中的每个事件
            for event_idx, event in enumerate(edl):
                logger.info(f"处理EDL事件 {event_idx + 1}/{len(edl)}: 类型 '{event.get('event_type', 'N/A')}'")
                
                audio_start_time = event.get("audio_start_time")
                audio_end_time = event.get("audio_end_time")
                audio_text = event.get("audio_text", "")
                
                logger.info(f"音频文本: '{audio_text[:30]}...' 时长: {audio_end_time - audio_start_time:.2f}s")

                if audio_start_time is None or audio_end_time is None:
                    logger.warning(f"由于缺少开始/结束时间，跳过事件 {event_idx + 1}")
                    continue
                
                segment_duration = audio_end_time - audio_start_time

                if segment_duration <= 0.001: # 使用一个小的epsilon
                    logger.info(f"由于时长为零或可忽略不计，跳过事件 {event_idx + 1}: {segment_duration:.3f}s")
                    continue
                
                total_expected_video_duration += segment_duration
                current_segment_parts = [] 
                
                # 处理新格式的video_segments
                video_segments = event.get("video_segments", [])
                
                if not video_segments:
                    logger.warning(f"事件 {event_idx + 1} 没有视频片段，使用黑屏")
                    black_clip = ColorClip(
                        size=(self.output_width, self.output_height),
                        color=(0, 0, 0),
                        duration=segment_duration,
                        ismask=False
                    )
                    all_moviepy_clips.append(black_clip)
                    current_segment_parts.append(black_clip)
                else:
                    # 跟踪此段已使用的时长
                    used_duration = 0.0
                    
                    for segment_idx, video_segment in enumerate(video_segments):
                        video_id = video_segment.get("video_id")
                        video_filename = video_segment.get("filename")
                        segment_duration_target = video_segment.get("duration", 0)
                        similarity_score = video_segment.get("similarity_score", 0)
                        
                        logger.info(f"  处理视频片段 {segment_idx + 1}: {video_filename}")
                        logger.info(f"  目标时长: {segment_duration_target:.2f}s, 相似度: {similarity_score:.4f}")
                        
                        # 在主文件夹中查找每个子文件夹
                        video_file_path = None
                        for folder_name in os.listdir(self.video_clips_base_dir):
                            folder_path = os.path.join(self.video_clips_base_dir, folder_name)
                            if not os.path.isdir(folder_path):
                                continue
                                
                            # 检查此视频是否属于这个子文件夹
                            if video_filename.startswith(folder_name):
                                potential_path = os.path.join(folder_path, video_filename)
                                logger.info(f"  检查子文件夹中的路径: {potential_path}")
                                
                                if os.path.exists(potential_path):
                                    video_file_path = potential_path
                                    logger.info(f"  找到视频文件: {video_file_path}")
                                    break
                        
                        # 如果在子文件夹中没找到，尝试直接在基础目录中查找
                        if not video_file_path:
                            direct_path = os.path.join(self.video_clips_base_dir, video_filename)
                            logger.info(f"  检查直接路径: {direct_path}")
                            if os.path.exists(direct_path):
                                video_file_path = direct_path
                                logger.info(f"  找到视频文件: {video_file_path}")
                        
                        # 如果找到视频文件，加载并处理它
                        if video_file_path and os.path.exists(video_file_path):
                            try:
                                # 使用详细日志跟踪视频加载过程
                                logger.info(f"  正在加载视频文件: {video_file_path}")
                                
                                # 打印文件大小信息
                                file_size = os.path.getsize(video_file_path)
                                logger.info(f"  文件大小: {file_size} 字节")
                                
                                # 尝试加载视频
                                b_roll_clip = VideoFileClip(video_file_path, audio=False)
                                all_moviepy_clips.append(b_roll_clip)
                                
                                # 记录加载的视频信息
                                logger.info(f"  加载成功! 视频尺寸: {b_roll_clip.size}, 时长: {b_roll_clip.duration}s, FPS: {b_roll_clip.fps}")
                                
                                # 处理第一帧，检查视频内容是否可读
                                try:
                                    first_frame = b_roll_clip.get_frame(0)
                                    mean_pixel_value = np.mean(first_frame)
                                    logger.info(f"  第一帧平均像素值: {mean_pixel_value:.2f} (0=黑色, 255=白色)")
                                except Exception as frame_err:
                                    logger.warning(f"  无法获取第一帧: {frame_err}")
                                
                                # 调整视频大小
                                resized_clip = b_roll_clip.fx(resize, width=self.output_width)
                                if resized_clip.size[1] > self.output_height:
                                    resized_clip = b_roll_clip.fx(resize, height=self.output_height)
                                if resized_clip is not b_roll_clip:
                                    all_moviepy_clips.append(resized_clip)
                                
                                logger.info(f"  调整大小后: {resized_clip.size}")
                                
                                # 创建背景
                                background = ColorClip(
                                    size=(self.output_width, self.output_height),
                                    color=(0, 0, 0),
                                    duration=resized_clip.duration,
                                    ismask=False
                                )
                                all_moviepy_clips.append(background)
                                
                                # 创建合成视频
                                composite_clip = CompositeVideoClip(
                                    [background, resized_clip.set_position("center")],
                                    size=(self.output_width, self.output_height)
                                )
                                all_moviepy_clips.append(composite_clip)
                                
                                # 检查时长是否需要裁剪
                                if composite_clip.duration > segment_duration_target:
                                    logger.info(f"  裁剪视频从 {composite_clip.duration:.2f}s 到 {segment_duration_target:.2f}s")
                                    clipped_video = composite_clip.subclip(0, segment_duration_target)
                                    all_moviepy_clips.append(clipped_video)
                                    current_segment_parts.append(clipped_video)
                                    used_duration += segment_duration_target
                                else:
                                    logger.info(f"  使用完整视频: {composite_clip.duration:.2f}s")
                                    current_segment_parts.append(composite_clip)
                                    used_duration += composite_clip.duration
                                
                            except Exception as e:
                                logger.error(f"  处理视频文件时出错: {e}", exc_info=True)
                                # 使用黑屏代替
                                logger.warning(f"  使用黑屏替代视频片段 {segment_idx + 1}")
                                black_clip = ColorClip(
                                    size=(self.output_width, self.output_height),
                                    color=(0, 0, 0),
                                    duration=segment_duration_target,
                                    ismask=False
                                )
                                all_moviepy_clips.append(black_clip)
                                current_segment_parts.append(black_clip)
                                used_duration += segment_duration_target
                        else:
                            logger.warning(f"  找不到视频文件: {video_filename}")
                            logger.warning(f"  尝试过的路径包括所有以'{video_filename.split('-', 1)[0] if '-' in video_filename else ''}' 开头的子文件夹")
                            # 使用黑屏代替
                            logger.warning(f"  使用黑屏替代视频片段 {segment_idx + 1}")
                            black_clip = ColorClip(
                                size=(self.output_width, self.output_height),
                                color=(0, 0, 0),
                                duration=segment_duration_target,
                                ismask=False
                            )
                            all_moviepy_clips.append(black_clip)
                            current_segment_parts.append(black_clip)
                            used_duration += segment_duration_target
                    
                    # 检查是否需要添加额外的黑屏填充
                    remaining_duration = segment_duration - used_duration
                    if remaining_duration > 0.001:
                        logger.info(f"  添加 {remaining_duration:.2f}s 的黑屏填充")
                        black_clip = ColorClip(
                            size=(self.output_width, self.output_height),
                            color=(0, 0, 0),
                            duration=remaining_duration,
                            ismask=False
                        )
                        all_moviepy_clips.append(black_clip)
                        current_segment_parts.append(black_clip)
                
                video_subclips_for_concatenation.extend(current_segment_parts)
            
            if not video_subclips_for_concatenation:
                logger.error("未成功生成任何视频片段。无法创建最终视频。")
                return False

            logger.info(f"连接 {len(video_subclips_for_concatenation)} 个视频片段...")
            final_video_composition = concatenate_videoclips(video_subclips_for_concatenation, method="compose")
            final_video_composition = final_video_composition.set_fps(self.output_fps)
            all_moviepy_clips.append(final_video_composition)

            actual_video_duration = final_video_composition.duration
            logger.info(f"总视频时长: {actual_video_duration:.2f}s. 预期时长: {total_expected_video_duration:.2f}s")

            # 准备音频
            final_narration_to_use = narration_clip
            if narration_clip.duration > actual_video_duration:
                logger.info(f"裁剪音频从 {narration_clip.duration:.2f}s 到视频时长 {actual_video_duration:.2f}s.")
                final_narration_to_use = narration_clip.subclip(0, actual_video_duration)
                all_moviepy_clips.append(final_narration_to_use)
            elif narration_clip.duration < actual_video_duration:
                logger.warning(f"旁白音频 ({narration_clip.duration:.2f}s) 短于合成视频 ({actual_video_duration:.2f}s). 视频末尾将没有声音。")
            
            if final_narration_to_use is narration_clip:
                all_moviepy_clips.append(narration_clip)

            final_video_with_audio = final_video_composition.set_audio(final_narration_to_use)
            all_moviepy_clips.append(final_video_with_audio)

            logger.info(f"写入最终视频到 {output_video_path} (编解码器: {self.video_codec}, 音频编解码器: {self.audio_codec}, FPS: {self.output_fps})")
            
            output_dir = os.path.dirname(output_video_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            temp_audio_for_muxing = os.path.join(output_dir if output_dir else ".", self.temp_audiofile_name)

            final_video_with_audio.write_videofile(
                output_video_path,
                codec=self.video_codec,
                audio_codec=self.audio_codec,
                fps=self.output_fps,
                preset=self.video_preset,
                threads=self.ffmpeg_threads,
                logger='bar',
                temp_audiofile=temp_audio_for_muxing
            )
            
            if os.path.exists(temp_audio_for_muxing):
                try: 
                    os.remove(temp_audio_for_muxing)
                    logger.debug(f"删除临时音频文件: {temp_audio_for_muxing}")
                except OSError as e_remove: 
                    logger.warning(f"无法删除临时音频文件 {temp_audio_for_muxing}: {e_remove}")

            logger.info("视频合成成功完成。")
            return True

        except Exception as e:
            logger.error(f"视频合成过程中发生错误: {e}", exc_info=True)
            return False
        finally:
            logger.debug(f"关闭 {len(all_moviepy_clips)} 个MoviePy对象。")
            closed_count = 0
            for i, clip_obj in enumerate(all_moviepy_clips):
                if clip_obj and hasattr(clip_obj, 'close') and callable(clip_obj.close):
                    try:
                        clip_obj.close()
                        closed_count +=1
                    except Exception as e_close:
                        logger.debug(f"关闭MoviePy对象 #{i} (类型: {type(clip_obj)}) 时出错: {e_close}", exc_info=False)
            logger.debug(f"MoviePy对象清理: 关闭了 {closed_count} 个对象。")



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if not MOVIEPY_AVAILABLE:
        logger.error("MoviePy is not available. Cannot run Synthesizer tests that create video/audio.")
    else:
        logger.info("--- Testing Synthesizer (MoviePy is available) ---")
        
        dummy_config_for_synth_auto = {
            "paths": {
                "input_video_dir": "temp_test_data/synth_videos_in_auto",
                "output_final_video_dir": "temp_test_data/synth_video_out_auto",
                "final_video_filename": "test_synth_output_auto.mp4"
            },
            "settings": { 
                "output_video_resolution": "auto", 
                "output_video_fps": 25,
                "video_codec": "libx264",
                "audio_codec": "aac",
                "video_preset": "ultrafast", 
                "ffmpeg_threads": 1, 
                "temp_audiofile_name": "test_temp_audio_auto.aac"
            }
        }

        base_test_dir_auto = "temp_test_data"
        clips_in_dir_auto = os.path.join(base_test_dir_auto, "synth_videos_in_auto")
        video_out_dir_auto = os.path.join(base_test_dir_auto, "synth_video_out_auto")
        os.makedirs(clips_in_dir_auto, exist_ok=True)
        os.makedirs(video_out_dir_auto, exist_ok=True)

        first_clip_width, first_clip_height = 640, 360 
        dummy_video_files_info_auto = {
            f"first_clip_{first_clip_width}x{first_clip_height}.mp4": {"duration": 5, "color": (255,0,0), "size": (first_clip_width, first_clip_height)},
            "clip2_green_auto.mp4": {"duration": 3, "color": (0,255,0), "size": (320,240)},
        }

        for fname, info in dummy_video_files_info_auto.items():
            fpath = os.path.join(clips_in_dir_auto, fname)
            if not os.path.exists(fpath):
                try:
                    ColorClip(size=info["size"],
                              color=info["color"], duration=info["duration"], ismask=False)\
                              .write_videofile(fpath, codec="libx264", preset="ultrafast", logger=None, audio=False, fps=dummy_config_for_synth_auto["settings"]["output_video_fps"]) # fps for dummy video
                    logger.info(f"Created dummy video for auto test: {fpath} with size {info['size']}")
                except Exception as e:
                    logger.error(f"Could not create dummy video {fpath}: {e}. Test might fail.")

        dummy_narration_path_auto = os.path.join(base_test_dir_auto, "dummy_narration_synth_test_auto.mp3")
        if not os.path.exists(dummy_narration_path_auto):
            try:
                import numpy as np
                from moviepy.audio.AudioClip import AudioArrayClip
                duration_audio = 7; audio_fps_test = 44100
                tone_freq = 220; t_audio = np.linspace(0, duration_audio, int(duration_audio * audio_fps_test), False)
                wave = 0.1 * np.sin(2 * np.pi * tone_freq * t_audio)
                audio_array = np.array([wave,wave]).T
                narration_audio_clip = AudioArrayClip(audio_array, fps=audio_fps_test)
                narration_audio_clip.write_audiofile(dummy_narration_path_auto, codec='mp3', logger=None)
                narration_audio_clip.close()
                logger.info(f"Created dummy narration audio for auto test: {dummy_narration_path_auto}")
            except Exception as e:
                logger.error(f"Could not create dummy narration {dummy_narration_path_auto}: {e}. Synthesis test might fail.")
        
        dummy_edl_auto = [
            {"event_type": "video_match", "audio_start_time": 0.0, "audio_end_time": 3.0, "matched_video_filename": f"first_clip_{first_clip_width}x{first_clip_height}.mp4", "needs_trimming": True},
            {"event_type": "video_match", "audio_start_time": 3.0, "audio_end_time": 5.0, "matched_video_filename": "clip2_green_auto.mp4", "needs_trimming": False},
            {"event_type": "slug", "audio_start_time": 5.0, "audio_end_time": 7.0} # Test a slug / black fill
        ]
        output_test_video_path_auto = os.path.join(video_out_dir_auto, dummy_config_for_synth_auto["paths"]["final_video_filename"])

        if os.path.exists(dummy_narration_path_auto) and os.path.exists(os.path.join(clips_in_dir_auto, f"first_clip_{first_clip_width}x{first_clip_height}.mp4")):
            synthesizer_instance_auto = Synthesizer(dummy_config_for_synth_auto)
            logger.info(f"Attempting to synthesize 'auto' resolution test video to: {output_test_video_path_auto}")
            success_auto = synthesizer_instance_auto.synthesize_video(dummy_edl_auto, dummy_narration_path_auto, output_test_video_path_auto)

            if success_auto:
                logger.info(f"Synthesizer 'auto' test successful. Output video at: {output_test_video_path_auto}")
                assert os.path.exists(output_test_video_path_auto)
                final_clip_check = None
                try:
                    final_clip_check = VideoFileClip(output_test_video_path_auto)
                    logger.info(f"Output video (auto) resolution: {final_clip_check.size}, FPS: {final_clip_check.fps}, Duration: {final_clip_check.duration:.2f}s")
                    assert final_clip_check.size[0] == first_clip_width
                    assert final_clip_check.size[1] == first_clip_height
                    assert final_clip_check.fps == dummy_config_for_synth_auto["settings"]["output_video_fps"]
                    # Expected duration is sum of EDL event durations (3s + 2s + 2s = 7s)
                    assert abs(final_clip_check.duration - 7.0) < 0.1 
                except Exception as e_check:
                    logger.error(f"Error checking 'auto' output video: {e_check}")
                finally:
                    if final_clip_check:
                        final_clip_check.close()
            else:
                logger.error("Synthesizer 'auto' test failed.")
        else:
            logger.warning("Dummy narration or first clip for 'auto' test not found. Skipping Synthesizer 'auto' functional test.")