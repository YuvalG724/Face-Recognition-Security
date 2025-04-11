#include <Arduino.h>
#include <ESP32Servo.h>
#include <WebServer.h>
const char* ssid="";
const char* password="";
WebServer server(80);
#define lock 13
Servo servo;
int lockpos=180;
int unlockpos=0;
void setup() {
  Serial.begin(115200);
  pinMode(lock,OUTPUT);
  servo.attach(lock,500,2400);
  servo.write(90);
  digitalWrite(lock,LOW);
  Serial.println("The project is starting");
  WiFi.begin(ssid,password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status()!=WL_CONNECTED){
    delay(1000);
    Serial.print(".");
  }
  Serial.println("WiFi connected!");
  Serial.println(WiFi.localIP());
  server.on("/unlock",[](){
    servo.write(unlockpos);
    Serial.println("Door is unlocked");
    delay(5000);
    servo.write(lockpos);
    Serial.println("Door is locked");
    delay(5000);
    server.send(200,"text/plain","Door Unlocked");
  });
  server.begin();
  Serial.println("ESP32 Web Server started");
}
void loop() {
  server.handleClient();
}
