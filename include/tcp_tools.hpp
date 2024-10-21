#ifndef TCP_TOOLS_HPP
#define TCP_TOOLS_HPP

#include "utils.hpp"
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>

extern Logger logger;

class TCP {
public:
  TCP() : mb_sock_connected(false) {}
  ~TCP() {
    if (mb_sock_connected) {
      close(m_sock);
      mb_sock_connected = false;
      logger.log(INFO, "Disconnected from TCP server.");
    }
  }

  int m_sock;
  bool mb_sock_connected;

  void connect_to_tcp(const std::string &ip, const int port) {
    m_sock = 0;
    struct sockaddr_in serv_addr;

    if ((m_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      logger.log(ERROR, "Socket creation failed.");
      return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
      logger.log(ERROR, "Invalid address / Address not supported.");
      return;
    }

    if (connect(m_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      logger.log(ERROR, "Connection Failed.");
      close(m_sock);
      return;
    }

    mb_sock_connected = true;
    logger.log(INFO, "Connected to TCP server at ", ip, ":", port);
    return;
  }
};
#endif
