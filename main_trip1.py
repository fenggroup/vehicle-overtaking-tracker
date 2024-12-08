from src.tracker import VehiclePassTracker

def main():
    tracker = VehiclePassTracker(
        image_sequence_path='./data/trip2/',
        start_frame=1700,
        end_frame=18425,
        output_folder='./results/trip1/'
    )
    tracker.process_sequence()

if __name__ == "__main__":
    main()