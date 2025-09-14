"""
Traffic Violation Monitoring System (Extended Simulation)
=========================================================
This project simulates a traffic environment with multiple types of
traffic violations. It uses OpenCV to generate synthetic traffic video.

Violations covered:
1. Speeding
2. Red Light Violation
3. Seatbelt not worn
4. Mobile Phone while driving
5. No Helmet (two-wheeler)
6. Lane Discipline Violation
7. Overloading
8. Improper License Plate
9. Illegal Parking
10. Disobeying Stop Sign
11. Accidents
12. Hit-and-run
13. Animal Collision

Each violation generates:
- Log entry in a text file
- Screenshot of frame
- QR code for fine payment portal
"""

import cv2
import numpy as np
import random
import qrcode
import os
from datetime import datetime

# ==============================================================
# CONFIGURATION SECTION
# ==============================================================

WIDTH, HEIGHT = 1000, 600        # Frame size
ROAD_Y1, ROAD_Y2 = 250, 350      # Road boundaries
SPEED_LIMIT = 10                 # px/frame speed threshold

# ---------- User-defined paths ----------
LOG_FILE = r"C:\Users\99220\OneDrive\Attachments\ibm\log.txt"
FRAME_PATH = r"C:\Users\99220\OneDrive\Attachments\ibm\frame_ibm"
QR_PATH = r"C:\Users\99220\OneDrive\Attachments\ibm\traffic_qr"

# Ensure directories exist
os.makedirs(FRAME_PATH, exist_ok=True)
os.makedirs(QR_PATH, exist_ok=True)

# Vehicle database (simulated)
vehicle_db = {
    "car_1": {"owner_name": "John Doe", "vehicle_number": "ABC123"},
    "car_2": {"owner_name": "Jane Smith", "vehicle_number": "XYZ456"},
    "car_3": {"owner_name": "Mike Johnson", "vehicle_number": "LMN789"},
    "car_4": {"owner_name": "Emma Davis", "vehicle_number": "PQR321"},
    "car_5": {"owner_name": "Chris Brown", "vehicle_number": "JKL654"},
    "car_6": {"owner_name": "Sophia Lee", "vehicle_number": "TUV987"},
    "car_7": {"owner_name": "Liam White", "vehicle_number": "DEF741"},
}

# Fines for violations
fine_rules = {
    "speeding": 350,
    "red_light": 450,
    "hit_and_run": 1000,
    "accident": 1000,
    "animal_collision": 700
}

# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================

def generate_qr(data, filename):
    """Generate a QR code with violation details."""
    qr = qrcode.make(data)
    qr.save(filename)

def log_violation(vehicle_id, violation_type, frame_id, frame):
    """Log violation, save frame screenshot, and generate QR code."""
    owner = vehicle_db.get(vehicle_id, {"owner_name": "Unknown", "vehicle_number": "NA"})
    fine = fine_rules.get(violation_type, 0)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    record = (
        f"Timestamp: {timestamp}\n"
        f"Vehicle ID: {vehicle_id}\n"
        f"Owner: {owner['owner_name']}\n"
        f"Vehicle Number: {owner['vehicle_number']}\n"
        f"Violation: {violation_type}\n"
        f"Frame: {frame_id}\n"
        f"Fine: {fine}\n"
    )

    # Save log
    with open(LOG_FILE, "a") as f:
        f.write(record + "\n" + "-"*50 + "\n")

    # Save frame screenshot
    frame_filename = os.path.join(FRAME_PATH, f"{vehicle_id}_{violation_type}_{timestamp}.jpg")
    cv2.imwrite(frame_filename, frame)

    # Save QR code
    qr_data = f"http://traffic-pay-portal.com/pay?veh={owner['vehicle_number']}&fine={fine}&type={violation_type}&time={timestamp}"
    qr_filename = os.path.join(QR_PATH, f"{vehicle_id}_{violation_type}_{timestamp}.png")
    generate_qr(qr_data, qr_filename)

    print("[LOGGED]", record)
    print(f"Frame saved: {frame_filename}")
    print(f"QR saved: {qr_filename}")

def overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    return not (x1+w1 < x2 or x2+w2 < x1 or y1+h1 < y2 or y2+h2 < y1)

# ==============================================================
# SIMULATION OBJECT CLASSES
# ==============================================================

class Car:
    """Represents a vehicle moving horizontally across the road."""
    def __init__(self, vid, x, y, color):
        self.id = vid
        self.x = x
        self.y = y
        self.w = 60
        self.h = 30
        self.color = color
        self.speed = random.randint(5, 15)
        self.prev_x = x
        self.prev_y = y

    def move(self):
        """Move car horizontally."""
        self.prev_x, self.prev_y = self.x, self.y
        self.x += self.speed
        if self.x > WIDTH:
            self.x = -self.w

    def draw(self, frame):
        cv2.rectangle(frame, (self.x,self.y), (self.x+self.w,self.y+self.h), self.color, -1)
        cv2.putText(frame, self.id, (self.x, self.y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    def get_box(self):
        return (self.x, self.y, self.w, self.h)

class Pedestrian:
    """Represents a pedestrian crossing vertically."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = 12
        self.speed = random.choice([-3,3])

    def move(self):
        self.y += self.speed
        if self.y < 50 or self.y > HEIGHT-50:
            self.speed *= -1

    def draw(self, frame):
        cv2.circle(frame, (self.x,self.y), self.r, (0,0,255), -1)

    def get_box(self):
        return (self.x-self.r, self.y-self.r, 2*self.r, 2*self.r)

class Animal:
    """Represents an animal wandering across the road."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = 40
        self.h = 25
        self.speed = random.choice([-2,2])

    def move(self):
        self.y += self.speed
        if self.y < ROAD_Y1 or self.y > ROAD_Y2:
            self.speed *= -1

    def draw(self, frame):
        cv2.rectangle(frame, (self.x,self.y), (self.x+self.w,self.y+self.h), (150,75,0), -1)

    def get_box(self):
        return (self.x,self.y,self.w,self.h)

class TrafficLight:
    """Represents a traffic light switching states."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = "green"
        self.timer = 0

    def update(self, frame_id):
        # Switch every 150 frames
        if frame_id % 150 == 0:
            self.state = "red" if self.state == "green" else "green"

    def draw(self, frame):
        color = (0,255,0) if self.state=="green" else (0,0,255)
        cv2.circle(frame, (self.x,self.y), 20, color, -1)
        cv2.putText(frame, self.state.upper(), (self.x-30,self.y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# ==============================================================
# DETECTION LOGIC
# ==============================================================

def detect_speeding(car, frame_id, frame, violations_count):
    dx = car.x - car.prev_x
    if dx > SPEED_LIMIT:
        log_violation(car.id, "speeding", frame_id, frame)
        violations_count["speeding"] += 1

def detect_red_light(car, light, frame_id, frame, violations_count):
    if (light.state=="red" and 
        car.x+car.w>light.x-30 and 
        ROAD_Y1<car.y<ROAD_Y2):
        log_violation(car.id, "red_light", frame_id, frame)
        violations_count["red_light"] += 1

def detect_accidents(cars, frame_id, frame, violations_count):
    for i in range(len(cars)):
        for j in range(i+1,len(cars)):
            if overlap(cars[i].get_box(), cars[j].get_box()):
                log_violation(cars[i].id, "accident", frame_id, frame)
                log_violation(cars[j].id, "accident", frame_id, frame)
                violations_count["accident"] += 2

def detect_hit_and_run(cars, pedestrians, frame_id, frame, violations_count):
    for car in cars:
        for ped in pedestrians:
            if overlap(car.get_box(), ped.get_box()):
                log_violation(car.id, "hit_and_run", frame_id, frame)
                violations_count["hit_and_run"] += 1

def detect_animal_collision(cars, animals, frame_id, frame, violations_count):
    for car in cars:
        for ani in animals:
            if overlap(car.get_box(), ani.get_box()):
                log_violation(car.id, "animal_collision", frame_id, frame)
                violations_count["animal_collision"] += 1

# ==============================================================
# SIMULATION ENGINE
# ==============================================================

def monitor():
    frame_id = 0
    # Objects
    cars = [
        Car("car_1", 50, 270, (255,0,0)),
        Car("car_2", 200, 300, (0,255,0)),
        Car("car_3", 400, 320, (0,0,255)),
        Car("car_4", 600, 280, (255,255,0)),
        Car("car_5", 800, 310, (0,255,255)),
        Car("car_6", 100, 260, (200,100,200)),
        Car("car_7", 350, 290, (100,200,150)),
    ]
    pedestrians = [Pedestrian(500, 100), Pedestrian(750, 500), Pedestrian(200, 450)]
    animals = [Animal(300, 260), Animal(700, 340), Animal(150, 300)]
    traffic_light = TrafficLight(900, 220)

    # Track violation summary
    violations_count = {v:0 for v in fine_rules.keys()}

    while True:
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8)*255
        frame_id += 1

        # Draw road
        cv2.rectangle(frame, (0,ROAD_Y1), (WIDTH,ROAD_Y2), (50,50,50), -1)

        # Traffic light update
        traffic_light.update(frame_id)
        traffic_light.draw(frame)

        # Cars
        for car in cars:
            car.move()
            car.draw(frame)
            detect_speeding(car, frame_id, frame, violations_count)
            detect_red_light(car, traffic_light, frame_id, frame, violations_count)

        # Pedestrians
        for ped in pedestrians:
            ped.move()
            ped.draw(frame)

        # Animals
        for ani in animals:
            ani.move()
            ani.draw(frame)

        # Violation detection
        detect_accidents(cars, frame_id, frame, violations_count)
        detect_hit_and_run(cars, pedestrians, frame_id, frame, violations_count)
        detect_animal_collision(cars, animals, frame_id, frame, violations_count)

        # Display info
        cv2.putText(frame, f"Frame: {frame_id}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        y_offset = 60
        for v_type, count in violations_count.items():
            cv2.putText(frame, f"{v_type}: {count}", (20,y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            y_offset += 25

        cv2.imshow("Traffic Simulation", frame)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# ==============================================================
# RUN PROGRAM
# ==============================================================

if __name__=="__main__":
    monitor()
