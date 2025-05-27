# src/main.py

import argparse
import logging
import os
import time
import shutil

from .utils.config_loader import load_config
from .utils.logger_setup import setup_logging
from .phase_1_analyzer import AudioAnalyzer, VideoAnalyzer
from .phase_2_matcher import Matcher
from .phase_3_timeline_generator import TimelineGenerator
from .phase_4_synthesizer import Synthesizer
from sentence_transformers import SentenceTransformer


# --- Constants ---
DEFAULT_CONFIG_PATH = "config.yaml"
def clear_output_folders(config, logger):
    """
    清空output文件夹中的edl和processed_data子文件夹。
    
    Args:
        config (dict): 配置字典
        logger (logging.Logger): 日志记录器
    """
    folders_to_clear = [
        config["paths"].get("output_processed_data_dir"),
        config["paths"].get("output_edl_dir")
    ]
    
    for folder in folders_to_clear:
        if folder and os.path.exists(folder):
            try:
                logger.info(f"正在清空文件夹: {folder}")
                # 删除文件夹中的所有文件，但保留文件夹本身
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        logger.warning(f"删除 {file_path} 时出错: {e}")
                logger.info(f"文件夹 {folder} 已清空")
            except Exception as e:
                logger.error(f"清空文件夹 {folder} 时出错: {e}")



def main():
    """
    Main function to orchestrate the AISmartMixer pipeline.
    """
    start_time = time.time()

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="AISmartMixer: AI-Powered Video Clip Assembler")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--force-reanalyze",
        action="store_true",
        help="Force re-analysis of audio and video even if intermediate files exist.",
    )

    parser.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip clearing output folders before processing.",
    )
    args = parser.parse_args()

    # --- Configuration Loading ---
    try:
        config = load_config(args.config)
        if config is None:
            # load_config already prints an error
            return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Override force_reanalyze from config if specified in CLI
    if args.force_reanalyze:
        config["force_reanalyze"] = True

    # --- Logging Setup ---
    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)
    logger.info("AISmartMixer application started.")
    logger.info(f"Using configuration file: {args.config}")
    if config.get("force_reanalyze"):
        logger.info("Force re-analyze is ENABLED.")

    # --- 清空输出文件夹 ---
    if not args.skip_clear:
        clear_output_folders(config, logger)
    else:
        logger.info("跳过清空输出文件夹")


    # --- Ensure Output Directories Exist ---
    try:
        os.makedirs(config["paths"]["output_processed_data_dir"], exist_ok=True)
        # output_edl_dir is used in config.yaml, ensure it's created
        os.makedirs(config["paths"].get("output_edl_dir", os.path.join(config["paths"]["output_processed_data_dir"], "edl")), exist_ok=True)
        os.makedirs(config["paths"]["output_final_video_dir"], exist_ok=True)
        logger.info("Output directories ensured.")
    except OSError as e:
        logger.error(f"Error creating output directories: {e}")
        return
    except KeyError as e:
        logger.error(f"Missing path configuration for directory creation: {e}")
        return


    # --- PHASE 1: ANALYSIS ---
    logger.info("Starting Phase 1: Analysis")
    # Audio Analysis
    audio_analyzer = AudioAnalyzer(config)
    audio_transcription_path = os.path.join(
        config["paths"]["output_processed_data_dir"],
        config["paths"]["audio_transcription_json"]
    )
    if not os.path.exists(audio_transcription_path) or config.get("force_reanalyze"):
        logger.info("Analyzing audio...")
        transcription_result = audio_analyzer.analyze()
        if transcription_result:
            audio_analyzer.save_transcription(transcription_result, audio_transcription_path)
            logger.info(f"Audio transcription saved to {audio_transcription_path}")
        else:
            logger.error("Audio analysis failed. Exiting.")
            return
    else:
        logger.info(f"Using existing audio transcription: {audio_transcription_path}")
        transcription_result = audio_analyzer.load_transcription(audio_transcription_path)
        if not transcription_result:
            logger.error(f"Failed to load existing transcription from {audio_transcription_path}. Consider re-analyzing.")
            return

    # Video Analysis
    embedding_model_name = config.get("models", {}).get("embedding_model_name", "all-MiniLM-L6-v2")
    embedding_model = None
    logger.info(f"Loading sentence transformer model for video analysis: {embedding_model_name}...")
    
    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Sentence transformer model '{embedding_model_name}' loaded successfully.")
    
    except Exception as e:
        logger.error(f"Failed to load sentence transformer model '{embedding_model_name}': {e}", exc_info=True)
        return


    video_analyzer = VideoAnalyzer(config, embedding_model) # Pass the loaded model
    video_metadata_path = os.path.join(
        config["paths"]["output_processed_data_dir"],
        config["paths"]["video_metadata_embeddings_json"]
    )
    if not os.path.exists(video_metadata_path) or config.get("force_reanalyze"):
        logger.info("Analyzing video clips...")
        video_metadata_list = video_analyzer.analyze_videos()
        if video_metadata_list:
            video_analyzer.save_video_metadata(video_metadata_list, video_metadata_path)
            logger.info(f"Video metadata and embeddings saved to {video_metadata_path}")
        else:
            logger.warning("No video clips were processed or video analysis failed.")
            video_metadata_list = [] # Ensure it's an empty list if analysis fails to return one
    else:
        logger.info(f"Using existing video metadata: {video_metadata_path}")
        video_metadata_list = video_analyzer.load_video_metadata(video_metadata_path)
        if not video_metadata_list:
            logger.warning(f"Failed to load existing video metadata from {video_metadata_path} or it was empty.")
            video_metadata_list = [] # Ensure it's an empty list


    if not transcription_result or not transcription_result.get("segments"):
        logger.error("No audio segments found after analysis. Cannot proceed.")
        return

    if not video_metadata_list: # This check is after ensuring video_metadata_list is at least an empty list
        logger.warning("No video metadata available. The final video might have missing visual content.")
        # Proceeding, but the matcher and timeline generator will handle the lack of video clips

    logger.info("Phase 1: Analysis completed.")

    # --- PHASE 2: MATCHER ---
    logger.info("Starting Phase 2: Matcher Initialization")
    # Matcher is initialized here, its methods will be used by TimelineGenerator
    matcher = Matcher(config)
    logger.info("Phase 2: Matcher Initialized.")


    # --- PHASE 3: TIMELINE GENERATOR (includes matching logic) ---
    logger.info("Starting Phase 3: Timeline Generation")
    timeline_generator = TimelineGenerator(config, matcher) # Pass the matcher instance
    
    # Path for the EDL/Timeline JSON, using key from config.yaml
    edl_output_dir = config["paths"].get("output_edl_dir", os.path.join(config["paths"]["output_processed_data_dir"], "edl"))
    timeline_json_filename = config["paths"].get("edit_decision_list_json", "final_edit_list.json") # As per config.yaml
    edit_decision_list_path = os.path.join(
        edl_output_dir,
        timeline_json_filename
    )


    edit_decision_list = timeline_generator.generate_timeline(
        transcription_result,
        video_metadata_list if video_metadata_list is not None else [] # Pass empty list if None
    )

    if edit_decision_list:
        # Corrected method call: save_edl -> save_timeline
        # The save_timeline method in TimelineGenerator handles the path logic internally if only filename is passed
        # or uses the full path if provided. Here we provide the full path.
        if timeline_generator.save_timeline(edit_decision_list, edit_decision_list_path):
            logger.info(f"Timeline (Edit Decision List) saved to {edit_decision_list_path}")
        else:
            logger.error(f"Failed to save Timeline (Edit Decision List) to {edit_decision_list_path}. Exiting.")
            return
    else:
        logger.error("Timeline generation failed or resulted in an empty EDL. Exiting.")
        return
    logger.info("Phase 3: Timeline Generation completed.")

    # --- PHASE 4: SYNTHESIZER ---
    logger.info("Starting Phase 4: Synthesizer")
    synthesizer = Synthesizer(config)
    final_video_path = os.path.join(
        config["paths"]["output_final_video_dir"],
        config["paths"]["final_video_filename"]
    )
    narration_audio_path = os.path.join(
        config["paths"]["input_audio_dir"],
        config["paths"]["input_audio_filename"]
    )

    success = synthesizer.synthesize_video(
        edit_decision_list,
        narration_audio_path,
        final_video_path
    )

    if success:
        logger.info(f"Final video successfully synthesized to {final_video_path}")
    else:
        logger.error("Video synthesis failed.")
    logger.info("Phase 4: Synthesizer completed.")

    # --- End of Process ---
    end_time = time.time()
    logger.info(f"AISmartMixer processing finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()