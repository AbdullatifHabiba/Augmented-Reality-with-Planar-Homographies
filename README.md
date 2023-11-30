# Video Processing App

Welcome to the Video Processing App! This web application allows users to upload two videos and an image for processing. The app is built using Flask and Docker for easy deployment.

## Features

- **User-friendly Interface**: A clean and intuitive interface for a seamless user experience.
- **File Upload**: Users can upload two video files (.mov) and an image file (.png) for processing.
- **Video Processing**: The uploaded videos and image are processed to generate a result.

## Prerequisites

Before running the app, ensure you have the following installed:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

## How to Use

1. Clone this repository to your local machine.

    ```bash
    git clone <repository-url>
    cd video-processing-app
    ```

2. Build and run the Docker containers.

    ```bash
    docker-compose up --build
    ```

3. Access the app in your web browser at `http://localhost:5000`.

4. Upload two video files and an image file for processing.

5. Click the "Process Videos" button to initiate the processing.

## Dependencies

The app relies on the following Python libraries:

- Flask
- OpenCV (opencv-python)
- NumPy
- Matplotlib

These dependencies are included in the `req.txt` file.

## Issues and Contributions

If you encounter any issues or have suggestions for improvement, please [open an issue](https://github.com/your-username/video-processing-app/issues) on GitHub.

Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
