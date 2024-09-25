#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

void splitYUV420sp(const unsigned char *inputImageData, int width, int height,
                   unsigned char *outputImageDatas[4]) {
  // Calculate sizes
  int ySize = width * height;
  int uvSize = ySize / 2; // Since UV are interleaved in YUV420sp

  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;

  int uvHalfWidth = yHalfWidth;
  int uvHalfHeight =
      yHalfHeight / 2; // UV plane height is half of Y plane height
  int uvHalfSize =
      uvHalfWidth * uvHalfHeight * 2; // *2 because UV are interleaved

  // Pointers to the Y and UV planes in the input image
  const unsigned char *yPlane = inputImageData;
  const unsigned char *uvPlane = inputImageData + ySize;

  // Process each quadrant
  for (int q = 0; q < 4; ++q) {
    // Calculate starting positions
    int yStartX = (q % 2) * yHalfWidth;
    int yStartY = (q / 2) * yHalfHeight;
    int uvStartX = yStartX;
    int uvStartY = yStartY / 2; // Because UV height is half of Y height

    // Pointers to the output Y and UV data
    unsigned char *yOutput = outputImageDatas[q];
    unsigned char *uvOutput = outputImageDatas[q] + yHalfSize;

    // Copy Y plane data
    for (int i = 0; i < yHalfHeight; ++i) {
      std::memcpy(yOutput + i * yHalfWidth,
                  yPlane + (yStartY + i) * width + yStartX, yHalfWidth);
    }

    // Copy UV plane data
    for (int i = 0; i < uvHalfHeight; ++i) {
      std::memcpy(uvOutput + i * uvHalfWidth,
                  uvPlane + (uvStartY + i) * width + uvStartX, uvHalfWidth);
    }
  }
}

void splitYUV420sp(const unsigned char *inputImageData, int width, int height,
                   std::vector<std::vector<unsigned char>> &outputImageDatas) {
  // Calculate sizes
  int ySize = width * height;
  int uvSize = ySize / 2; // Since UV are interleaved in YUV420sp

  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;

  int uvHalfWidth = yHalfWidth;
  int uvHalfHeight =
      yHalfHeight / 2; // UV plane height is half of Y plane height
  int uvHalfSize =
      uvHalfWidth * uvHalfHeight * 2; // *2 because UV are interleaved

  // Pointers to the Y and UV planes in the input image
  const unsigned char *yPlane = inputImageData;
  const unsigned char *uvPlane = inputImageData + ySize;

  // Process each quadrant
  for (int q = 0; q < 4; ++q) {
    // Calculate starting positions
    int yStartX = (q % 2) * yHalfWidth;
    int yStartY = (q / 2) * yHalfHeight;
    int uvStartX = yStartX;
    int uvStartY = yStartY / 2; // Because UV height is half of Y height

    // Pointers to the output Y and UV data
    unsigned char *yOutput = outputImageDatas[q].data();
    unsigned char *uvOutput = outputImageDatas[q].data() + yHalfSize;

    // Copy Y plane data
    for (int i = 0; i < yHalfHeight; ++i) {
      std::memcpy(yOutput + i * yHalfWidth,
                  yPlane + (yStartY + i) * width + yStartX, yHalfWidth);
    }

    // Copy UV plane data
    for (int i = 0; i < uvHalfHeight; ++i) {
      std::memcpy(uvOutput + i * uvHalfWidth,
                  uvPlane + (uvStartY + i) * width + uvStartX, uvHalfWidth);
    }
  }
}

int main() {
  // Example usage

  // Input image dimensions
  int width = 640;  // Replace with your image width
  int height = 640; // Replace with your image height

  // Calculate input image size
  int ySize = width * height;
  int uvSize = ySize / 2;
  int imageSize = ySize + uvSize;

  // Allocate memory for input image data
  unsigned char *inputImageData = new unsigned char[imageSize];

  // Load your image data into inputImageData
  std::ifstream inputFile("../data/dog_bike_car_640x640_yuv420sp.bin");
  if (!inputFile) {
    std::cerr << "error opening input file" << std::endl;
    delete[] inputImageData;
    return 1;
  }
  inputFile.read(reinterpret_cast<char *>(inputImageData), imageSize);
  inputFile.close();

  // Allocate memory for output images
  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;
  int uvHalfSize =
      yHalfSize / 2; // Since UV plane size is half of Y plane in YUV420sp
  int outputImageSize =
      yHalfSize + uvHalfSize * 2; // *2 because UV data is interleaved

  // test API 2:
  std::vector<std::vector<unsigned char>> v_outputImageData(
      4, std::vector<unsigned char>(outputImageSize));
  splitYUV420sp(inputImageData, width, height, v_outputImageData);

  // Use the output image data
  for (int i = 0; i < 4; ++i) {
    std::ofstream outputFile("split_yuv_" + std::to_string(i) + "_v2.bin",
                             std::ios::binary);
    if (!outputFile) {
      std::cerr << "Error opening output file" << std::endl;
      continue;
    }
    outputFile.write(reinterpret_cast<char *>(v_outputImageData[i].data()),
                     outputImageSize);
    outputFile.close();
  }

  // test API 1:
  unsigned char *outputImageDatas[4];
  for (int i = 0; i < 4; ++i) {
    outputImageDatas[i] = new unsigned char[outputImageSize];
  }

  // Call the split function
  splitYUV420sp(inputImageData, width, height, outputImageDatas);
  // Use the output image data
  for (int i = 0; i < 4; ++i) {
    std::ofstream outputFile("split_yuv_" + std::to_string(i) + ".bin",
                             std::ios::binary);
    if (!outputFile) {
      std::cerr << "Error opening output file" << std::endl;
      continue;
    }
    outputFile.write(reinterpret_cast<char *>(outputImageDatas[i]),
                     outputImageSize);
    outputFile.close();
  }


  // Clean up
  delete[] inputImageData;
  for (int i = 0; i < 4; ++i) {
    delete[] outputImageDatas[i];
  }

  return 0;
}
