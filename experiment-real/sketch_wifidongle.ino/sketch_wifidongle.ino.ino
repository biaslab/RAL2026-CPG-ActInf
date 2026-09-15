// Raw TCP <-> Serial bridge for the Bittle WiFi dongle (ESP8266).
//
// Petoi's stock sketch is an HTTP server: it forwards a command token to the
// NyBoard but never sends the board's reply back, so it cannot read the IMU.
// The experiment needs full duplex, so this sketch is a dumb pipe instead --
// bytes from the socket go to the board, bytes from the board go to the socket,
// and nothing interprets either direction. PetoiRobot's ardSerial protocol then
// works over TCP exactly as it does over USB.
//
// AP mode is deliberate: the dongle is its own access point, so its address is
// always 192.168.4.1 and never moves with a DHCP lease. Connect the laptop to
// "Bittle-AP", then:  python run_experiment.py --wifi --mode rate
//
// NOTE: do NOT echo received bytes back to the client. ardSerial's
// printSerialMessage() reads lines until one equals the command token, so an
// echo of "v\n" is mistaken for the board's ack and the IMU line that follows
// is dropped -- every read then parses to nothing and the attitude feedback
// silently dies.

#include <ESP8266WiFi.h>

WiFiServer server(23);
WiFiClient client;

void setup() {
  Serial.begin(115200);              // match OpenCat
  WiFi.mode(WIFI_AP);
  WiFi.softAP("Bittle-AP");          // laptop connects here; dongle is 192.168.4.1
  server.begin();
  server.setNoDelay(true);
}

void loop() {
  if (server.hasClient()) {
    if (!client || !client.connected()) {
      client = server.available();
      client.setNoDelay(true);
    } else server.available().stop();
  }
  // socket -> board
  while (client && client.available() && Serial.availableForWrite()) {
    Serial.write(client.read());
  }
  // board -> socket
  while (Serial.available() && client && client.connected()) {
    client.write(Serial.read());
  }
}
