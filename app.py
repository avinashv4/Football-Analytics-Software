import streamlit as st
import cv2
import numpy as np
import tempfile
import subprocess

from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os


def process_video(input_video_path):
    # Read the uploaded video file
    video_frames = read_video(input_video_path)

    # Initialize Tracker
    tracker = Tracker('best.pt')

    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    tracker.add_position_to_tracks(tracks)

    # Camera movement estimation
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View transformation
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimation
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Ball assignment to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # Annotate video with tracking info
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw speed and distance annotations
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the output video temporarily
    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v as a fallback
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = output_temp_file.name

    # Create VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    # Release everything if job is finished
    out.release()

    # Explicitly provide the path to ffmpeg.exe
    ffmpeg_path = r'C:\ffmpeg\ffmpeg-2024-10-02-git-358fdf3083-full_build\bin\ffmpeg.exe'  # Adjust this path
    converted_video_path = f"./converted_{os.path.basename(output_video_path)}"

    # Using subprocess to call FFmpeg with full path
    subprocess.call([ffmpeg_path, '-y', '-i', output_video_path, '-c:v', 'libx264', converted_video_path])
    
    return converted_video_path


def main():
    st.title("⚽ Football Video Analysis App")
    st.write("Upload a football video and get a detailed analysis of player positions, speed, ball control, and more! 📊")

    # Upload video section
    uploaded_file = st.file_uploader("🎥 Upload Your Football Video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.write("Processing your video... ⏳ This may take a while.")
        
        # Save the uploaded video file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        # Process the uploaded video
        output_video_path = process_video(temp_video_path)

        # Display the processed video
        st.write("🎬 Here's your analyzed video:")
        
        # Attempt to play the video in the app
        try:
            st.video(output_video_path)
        except:
            st.warning("⚠️ Unable to play the video. Please download it below.")

        # Provide download link for the processed video
        with open(output_video_path, 'rb') as video_file:
            video_bytes = video_file.read()

            # Display the download button
            st.download_button(
                label="⬇️ Download the Processed Video",
                data=video_bytes,
                file_name=os.path.basename(output_video_path),
                mime='video/mp4'
            )

        st.success("✅ Processing complete!")

    st.write("Feel free to analyze another video by uploading a new one! 🔄")

    # Footer
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #000;
            text-align: center;
            padding: 10px;
            font-size: small;
            color: #fff;
        }
        </style>
        <div class="footer">
            made with love by Avinash ❤️
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
