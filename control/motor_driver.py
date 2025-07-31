class MotorController:
    def __init__(self):
        
        pass #Placeholder for now to set up pins later

    def execute_action(self, action):
        if action == 0:
            self.move_forward()
        elif action == 1:
            self.turn_left()
        elif action == 2:
            self.turn_right()
        else:
            self.stop()

#Placeholder strings to later be replaced with pins
    def move_forward(self):
        print("Moving forward")

    def turn_left(self):
        print("Turning left")

    def turn_right(self):
        print("Turning right")

    def stop(self):
        print("Stopping")
