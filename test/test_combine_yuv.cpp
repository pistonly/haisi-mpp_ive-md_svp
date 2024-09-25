#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

void combine_YUV420sp(
    const std::vector<std::vector<unsigned char>> &v_yuv420sp_4, int width,
    int height, std::vector<unsigned char> &yuv420sp_combined) {
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

  // Pointers to the Y and UV planes in the target image
  unsigned char *yPlane = yuv420sp_combined.data();
  unsigned char *uvPlane = yPlane + ySize;

  // Process each quadrant
  for (int q = 0; q < 4; ++q) {
    // Calculate starting positions
    int yStartX = (q % 2) * yHalfWidth;
    int yStartY = (q / 2) * yHalfHeight;
    int uvStartX = yStartX;
    int uvStartY = yStartY / 2; // Because UV height is half of Y height

    // Pointers to the output Y and UV data
    const unsigned char *ySrc = v_yuv420sp_4[q].data();
    const unsigned char *uvSrc = v_yuv420sp_4[q].data() + yHalfSize;

    // Copy Y plane data
    for (int i = 0; i < yHalfHeight; ++i) {
      std::memcpy(yPlane + (yStartY + i) * width + yStartX,
                  ySrc + i * yHalfWidth, yHalfWidth);
    }

    // Copy UV plane data
    for (int i = 0; i < uvHalfHeight; ++i) {
      std::memcpy(uvPlane + (uvStartY + i) * width + uvStartX,
                  uvSrc + i * uvHalfWidth, uvHalfWidth);
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

  int yHalfWidth = width / 2;
  int yHalfHeight = height / 2;
  int yHalfSize = yHalfWidth * yHalfHeight;
  int uvHalfSize =
      yHalfSize / 2; // Since UV plane size is half of Y plane in YUV420sp
  int smallImageSize =
      yHalfSize + uvHalfSize * 2; // *2 because UV data is interleaved

  // read input image
  std::vector<std::vector<unsigned char>> yuv_smalls(
      4, std::vector<unsigned char>(smallImageSize));
  for (int i = 0; i < 4; ++i) {
    std::ifstream inputFile("split_yuv_" + std::to_string(i) + "_v2.bin",
                            std::ios::binary);
    inputFile.read(reinterpret_cast<char *>(yuv_smalls[i].data()),
                   smallImageSize);
    inputFile.close();
  }

  // combine
  std::vector<unsigned char> yuv_large(imageSize);
  combine_YUV420sp(yuv_smalls, width, height, yuv_large);

  // save image
  std::ofstream outputFile("combined_yuv.bin", std::ios::binary);
  if (!outputFile) {
    std::cerr << "Error opening output file" << std::endl;
    return 1;
  }
  outputFile.write(reinterpret_cast<char *>(yuv_large.data()), imageSize);
  outputFile.close();

  return 0;
}
