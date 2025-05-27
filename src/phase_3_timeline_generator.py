# src/phase_3_timeline_generator.py

import logging
import os

from .utils.file_handler import save_json
# We don't directly use Matcher here in __main__, but it's a dependency for the class
# from .phase_2_matcher import Matcher # Needed if running __main__ with full objects

logger = logging.getLogger(__name__)

class TimelineGenerator:
    """
    Generates a timeline by matching audio segments to video clips.
    """
    def __init__(self, config, matcher):
        """
        初始化TimelineGenerator。
        
        Args:
            config (dict): 应用程序配置字典。
            matcher (Matcher): Matcher类的实例。
        """
        self.config = config
        self.matcher = matcher
        if not self.matcher or not hasattr(self.matcher, 'find_best_matches'):
            logger.error("TimelineGenerator: 提供的Matcher实例无效或未初始化。")
            raise ValueError("TimelineGenerator需要有效的Matcher实例。")

        self.timeline_options = self.config.get("timeline", {})
        self.min_segment_duration_for_match = self.timeline_options.get("min_segment_duration_for_match", 0.5)
        
        # 初始化全局已使用视频集合
        self.globally_used_videos = set()


    def _select_videos_for_segment(self, audio_segment, video_metadata_map, segment_duration):
        """
        为音频段选择最佳视频片段组合，避免重复使用已使用过的视频。
        
        Args:
            audio_segment (dict): 要匹配的音频段。
            video_metadata_map (dict): 视频ID到视频元数据的映射。
            segment_duration (float): 需要填充的总时长。
        
        Returns:
            list: 包含(video_id, score, start_time, duration)元组的列表，表示如何拼接视频。
        """
        # 筛选可用视频
        available_videos = []
        for video_id, video_meta in video_metadata_map.items():
            if video_id not in self.globally_used_videos:
                available_videos.append(video_meta)
        
        if not available_videos:
            logger.warning(f"没有可用的未使用视频片段，无法为音频段 {audio_segment.get('id')} 生成视频")
            return []
        
        # 获取候选视频（根据相似度排序）
        potential_matches = self.matcher.find_best_matches(audio_segment, available_videos, top_n=15)
        if not potential_matches:
            return []
        
        # 结果列表
        selected_videos = []
        remaining_duration = segment_duration
        
        # 尝试填充所需时长
        for video_id, score, video_duration in potential_matches:
            if remaining_duration <= 0:
                break
            
            if video_id in self.globally_used_videos:
                continue
            
            # 视频时长实际可能与候选结果不同，使用视频元数据中的时长
            actual_duration = video_metadata_map.get(video_id, {}).get("duration", video_duration)
            if actual_duration is None or actual_duration <= 0:
                logger.warning(f"视频 {video_id} 没有有效的时长信息，跳过")
                continue
            
            # 将视频添加到选择列表
            use_duration = min(actual_duration, remaining_duration)
            selected_videos.append((video_id, score, 0, use_duration))
            remaining_duration -= use_duration
            
            # 标记为已使用
            self.globally_used_videos.add(video_id)
            
            if remaining_duration <= 0:
                break
        
        return selected_videos

    def generate_timeline(self, audio_transcription, video_metadata_list):
        """
        生成时间线事件序列。
        
        Args:
            audio_transcription (dict): 带有片段的解析音频转录。
            video_metadata_list (list): 视频元数据字典列表。
        
        Returns:
            list: 时间线事件字典列表。
        """
        # 重置全局已使用视频跟踪
        self.globally_used_videos = set()
        
        if not audio_transcription or "segments" not in audio_transcription:
            logger.error("提供的音频转录数据无效或为空。")
            return []
        if not video_metadata_list:
            logger.error("视频元数据列表为空。无法生成时间线。")
            return []

        # 将video_metadata_list转换为便于按ID查找的映射
        video_metadata_map = {vm["id"]: vm for vm in video_metadata_list if "id" in vm}
        if not video_metadata_map:
            logger.error("视频元数据列表不包含具有'id'的有效项目。")
            return []

        timeline_events = []
        sorted_audio_segments = sorted(audio_transcription["segments"], key=lambda s: s["start"])

        for i, segment in enumerate(sorted_audio_segments):
            segment_id = segment.get("id", f"segment_{i}")
            segment_text = segment.get("text", "").strip()
            segment_duration = segment.get("end", 0) - segment.get("start", 0)

            if not segment_text or segment_duration < self.min_segment_duration_for_match:
                logger.info(f"跳过音频段 ID {segment_id}，因为文本为空或时长过短 ({segment_duration:.2f}s)。")
                timeline_events.append({
                    "event_type": "slug",
                    "audio_segment_id": segment_id,
                    "audio_start_time": segment.get("start"),
                    "audio_end_time": segment.get("end"),
                    "reason": "文本为空或过短"
                })
                continue

            logger.info(f"处理音频段 ID {segment_id}: '{segment_text[:50]}...'")
            
            # 为音频段选择多个视频片段
            selected_videos = self._select_videos_for_segment(
                segment, video_metadata_map, segment_duration
            )
            
            # 创建包含多个视频的事件
            event = {
                "event_type": "video_match",
                "audio_segment_id": segment_id,
                "audio_start_time": segment.get("start"),
                "audio_end_time": segment.get("end"),
                "audio_text": segment_text,
                "video_segments": []  # 包含多个视频片段
            }
            
            if not selected_videos:
                event["event_type"] = "match_failed"
                event["notes"] = "没有匹配的视频可用"
                timeline_events.append(event)
                logger.warning(f"  无法为音频段 ID {segment_id} 找到视频匹配")
                continue
            
            total_duration = 0
            for video_id, score, start_time, duration in selected_videos:
                video_meta = video_metadata_map.get(video_id)
                video_segment = {
                    "video_id": video_id,
                    "filename": video_meta.get("filename", "unknown"),
                    "similarity_score": round(score, 4),
                    "start_time": start_time,
                    "duration": duration
                }
                event["video_segments"].append(video_segment)
                total_duration += duration
                
                logger.info(f"  为音频段 {segment_id} 分配视频 {video_id} (分数: {score:.4f}, 时长: {duration:.2f}s)")
            
            if total_duration < segment_duration:
                # 如果所有可用视频都不足以填充音频段
                logger.warning(f"  所有可用视频总时长 ({total_duration:.2f}s) 小于音频段时长 ({segment_duration:.2f}s)，将使用黑屏填充剩余部分")
                event["needs_black_fill"] = True
                event["black_fill_duration"] = segment_duration - total_duration
            
            timeline_events.append(event)

        logger.info(f"时间线生成完成。创建了 {len(timeline_events)} 个事件。")
        return timeline_events


    def save_timeline(self, timeline_data, output_filename=None):
        """
        Saves the generated timeline to a JSON file.

        Args:
            timeline_data (list): The list of timeline events.
            output_filename (str, optional): The name of the output file.
                If None, uses the path from the configuration.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if output_filename is None:
            processed_dir = self.config.get("paths", {}).get("output_processed_data_dir", "data/processed_data")
            filename = self.config.get("paths", {}).get("timeline_json", "timeline.json")
            output_path = os.path.join(processed_dir, filename)
        else: # If a specific filename (which might be a full path) is given
            if os.path.isabs(output_filename) or os.path.dirname(output_filename): # if it's a full path or relative path with dir
                output_path = output_filename
            else: # it's just a filename, use processed_dir
                processed_dir = self.config.get("paths", {}).get("output_processed_data_dir", "data/processed_data")
                output_path = os.path.join(processed_dir, output_filename)


        logger.info(f"Attempting to save timeline to: {output_path}")
        return save_json(timeline_data, output_path)


if __name__ == "__main__":
    # Setup basic logging for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # --- Mocking dependencies (Matcher) ---
    class MockMatcher:
        def __init__(self, config_dict):
            self.model_name = config_dict.get("matching",{}).get("embedding_model")
            self.threshold = config_dict.get("matching",{}).get("similarity_threshold", 0.1)
            logger.info(f"MockMatcher initialized with model {self.model_name} and threshold {self.threshold}")

        def find_best_matches(self, audio_segment, video_metadata_list, top_n=5):
            logger.debug(f"MockMatcher: find_best_matches called for audio text: '{audio_segment['text'][:30]}...'")
            # Simulate some matching logic based on keywords for testing
            matches = []
            audio_text_lower = audio_segment['text'].lower()
            for video_meta in video_metadata_list:
                video_desc_lower = video_meta.get('description_for_embedding', video_meta.get('filename','')).lower()
                score = 0.0
                if "park" in audio_text_lower and "park" in video_desc_lower: score = 0.9
                elif "car" in audio_text_lower and "car" in video_desc_lower: score = 0.85
                elif "ocean" in audio_text_lower and "ocean" in video_desc_lower: score = 0.92
                elif "cat" in video_desc_lower: score = 0.5 # Generic match
                else: score = 0.2 # Low score for others

                if score >= self.threshold:
                    matches.append((video_meta['id'], score))
            
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:top_n]

    # --- Dummy Config for Testing ---
    dummy_config_for_timeline = {
        "paths": {
            "output_processed_data_dir": "temp_test_data/processed",
            "timeline_json": "test_timeline.json"
        },
        "matching": { # Needed for MockMatcher
             "embedding_model": "mock_model",
             "similarity_threshold": 0.3
        },
        "timeline": {
            "allow_video_repetition": False,
            "max_video_uses": 1, # Irrelevant if allow_video_repetition is False
            "default_video_id_if_no_match": "vid_default",
            "min_segment_duration_for_match": 0.1,
            "num_candidates_for_selection": 5
        }
    }
    os.makedirs(dummy_config_for_timeline["paths"]["output_processed_data_dir"], exist_ok=True)


    logger.info("--- Testing TimelineGenerator ---")
    mock_matcher_instance = MockMatcher(dummy_config_for_timeline)
    timeline_generator = TimelineGenerator(dummy_config_for_timeline, mock_matcher_instance)

    # --- Test Data ---
    test_audio_transcription = {
        "segments": [
            {"id": "seg_001", "start": 0.0, "end": 5.0, "text": "A beautiful sunny day in the park with children playing."},
            {"id": "seg_002", "start": 5.5, "end": 10.0, "text": "A fast car chase through the city streets at night."},
            {"id": "seg_003", "start": 10.5, "end": 15.0, "text": "A quiet moment by the ocean waves."},
            {"id": "seg_004", "start": 15.5, "end": 16.0, "text": "Short."}, # Test min_segment_duration
            {"id": "seg_005", "start": 16.5, "end": 20.0, "text": "Something completely different without direct video match."},
            {"id": "seg_006", "start": 20.5, "end": 25.0, "text": "Another day in the park, perhaps a picnic."} # Test repetition
        ]
    }
    test_video_metadata = [
        {"id": "vid_001", "filename": "park_footage_01.mp4", "description_for_embedding": "People enjoying a sunny park, kids on swings."},
        {"id": "vid_002", "filename": "car_chase_scene.mp4", "description_for_embedding": "High-speed car pursuit in urban setting."},
        {"id": "vid_003", "filename": "ocean_sunset.mp4", "description_for_embedding": "Serene beach with gentle ocean waves."},
        {"id": "vid_004", "filename": "random_cat_video.mp4", "description_for_embedding": "A fluffy cat playing with a toy."},
        {"id": "vid_default", "filename": "default_placeholder.mp4", "description_for_embedding": "Default placeholder screen."}
    ]

    # --- Generate Timeline ---
    generated_timeline = timeline_generator.generate_timeline(test_audio_transcription, test_video_metadata)

    if generated_timeline:
        logger.info("\n--- Generated Timeline Events: ---")
        for event_idx, event in enumerate(generated_timeline):
            logger.info(f"Event {event_idx + 1}:")
            logger.info(f"  Audio Segment ID: {event['audio_segment_id']}")
            logger.info(f"  Audio Text: '{event['audio_text'][:40]}...'")
            logger.info(f"  Event Type: {event['event_type']}")
            if event.get("matched_video_id"):
                logger.info(f"  Matched Video ID: {event['matched_video_id']} (Filename: {event['matched_video_filename']})")
                logger.info(f"  Similarity Score: {event['similarity_score']}")
            if event.get("notes"):
                logger.info(f"  Notes: {event['notes']}")

        # --- Assertions for basic correctness (example) ---
        assert len(generated_timeline) == len(test_audio_transcription["segments"])
        # seg_001 should match vid_001
        assert generated_timeline[0]["matched_video_id"] == "vid_001"
        # seg_002 should match vid_002
        assert generated_timeline[1]["matched_video_id"] == "vid_002"
        # seg_004 should be a slug
        assert generated_timeline[3]["event_type"] == "slug"
        # seg_005 should use default video if no direct match by mock logic
        assert generated_timeline[4]["matched_video_id"] == "vid_default" or generated_timeline[4]["matched_video_id"] == "vid_004" # Mock might pick cat
        # seg_006 (park again) should use default if vid_001 (park) cannot be repeated
        # (This depends on allow_video_repetition = False and if vid_001 was used for seg_001)
        if not dummy_config_for_timeline["timeline"]["allow_video_repetition"]:
             assert generated_timeline[5]["matched_video_id"] == "vid_default" or generated_timeline[5]["matched_video_id"] == "vid_004", \
                 f"Expected default or another video for seg_006 due to no-repetition, got {generated_timeline[5]['matched_video_id']}"


        # --- Test Saving Timeline ---
        if timeline_generator.save_timeline(generated_timeline):
            saved_path = os.path.join(dummy_config_for_timeline["paths"]["output_processed_data_dir"], dummy_config_for_timeline["paths"]["timeline_json"])
            logger.info(f"\nTimeline successfully saved to {saved_path}")
            assert os.path.exists(saved_path)
        else:
            logger.error("\nFailed to save the timeline.")
    else:
        logger.error("Timeline generation failed or produced an empty timeline.")

    logger.info("\nTimelineGenerator tests completed.")
    # Consider cleaning up temp_test_data