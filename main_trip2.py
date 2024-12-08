from src.tracker import VehiclePassTracker

def main():
    tracker = VehiclePassTracker(
        image_sequence_path='./data/trip2/',
        start_frame=1800,
        end_frame=27466,
        output_folder='./results/trip2/'
    )
    tracker.process_sequence()

if __name__ == "__main__":
    main()