
#include <iostream>
#include <vector>
#include <half.hpp> // Half库，用于处理float16类型
#include <fstream>

using half_float::half;

// 将 std::vector<std::vector<char>> 解析为 std::vector<std::vector<half>>
std::vector<std::vector<half>> convertToHalf(const std::vector<std::vector<char>>& m_outputs) {
    std::vector<std::vector<half>> result;

    for (const auto& row : m_outputs) {
        std::vector<half> float_row;
        for (size_t i = 0; i < row.size(); i += 2) {
            // 将两个字节解析为一个float16（half）
            unsigned short half_value = static_cast<unsigned char>(row[i]) |
                                        (static_cast<unsigned char>(row[i + 1]) << 8);
            float_row.push_back(*reinterpret_cast<half*>(&half_value));
        }
        result.push_back(float_row);
    }

    return result;
}

int main() {
    // 示例：std::vector<std::vector<char>> 中存储的是float16的数据
    std::vector<std::vector<char>> m_outputs = {
        {'\x00', '\x3C', '\x00', '\x3C'},  // 两个 float16 的数值
        {'\x00', '\x3C', '\x00', '\x3C'}
    };

    // 转换为 float16（half） 类型
    std::vector<std::vector<half>> half_matrix = convertToHalf(m_outputs);

    // 输出结果
    for (const auto& row : half_matrix) {
        for (const auto& value : row) {
            std::cout << static_cast<float>(value) << " ";  // 将 half 转换为 float
        }
        std::cout << std::endl;
    }

    // method 2
    std::vector<std::vector<half>> half_matrix2 = reinterpret_cast<std::vector<std::vector<half>> &>(m_outputs);
    for (const auto & row: half_matrix2) {
      for (const auto& value : row) {
        std::cout << static_cast<float>(value) << " ";
      }
      std::cout << std::endl;
    }

    // test on python
    std::ofstream outFile("float16.bin", std::ios::binary);
    if (!outFile){
      std::cerr << "error opening file for writing!" << std::endl;
      return 1;
    }

    outFile.write(reinterpret_cast<const char*>(m_outputs[0].data()), m_outputs[0].size() * sizeof(char));

    return 0;
}
