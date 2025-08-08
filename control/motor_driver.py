from gpiozero import OutputDevice
from time import sleep

class MotorController:
    def __init__(self):
        # Motor A (Left) pins
        self.in1 = OutputDevice(22)  # GPIO22 (IN1)
        self.in2 = OutputDevice(23)  # GPIO23 (IN2)

        # Motor B (Right) pins
        self.in3 = OutputDevice(25)  # GPIO25 (IN3)
        self.in4 = OutputDevice(24)  # GPIO24 (IN4)

    def execute_action(self, action):
        if action == 0:
            self.move_forward()
        elif action == 1:
            self.turn_left()
        elif action == 2:
            self.turn_right()
        else:
            self.stop()

    def move_forward(self):
        print("Moving forward")
        # Both motors forward
        self.in1.on()
        self.in2.off()
        self.in3.on()
        self.in4.off()

    def turn_left(self):
        print("Turning left")
        # Left motor backward, right motor forward
        self.in1.off()
        self.in2.on()
        self.in3.on()
        self.in4.off()

    def turn_right(self):
        print("Turning right")
        # Left motor forward, right motor backward
        self.in1.on()
        self.in2.off()
        self.in3.off()
        self.in4.on()

    def stop(self):
        print("Stopping")
        # All motors stop
        self.in1.off()
        self.in2.off()
        self.in3.off()
        self.in4.off()
