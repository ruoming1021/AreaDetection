import cv2

class Camera:
    def __init__(self, cam_preset_num=15):
        self.cam_preset_num = cam_preset_num

    def get_cam_num(self):
        cnt = 0
        num = 0
        for device in range(0, self.cam_preset_num):            
            stream = cv2.VideoCapture(device)            
            if stream.isOpened():
                print(device)
                ret, frame = stream.read()
                cv2.imshow("stream",frame)
                cv2.imwrite("/home/vip/Desktop/yolov7/"+str(num)+".jpg",frame)
                cv2.waitKey(5000)
                num = num + 1
                grabbed = stream.grab()
                stream.release()
                
                cnt = cnt + 1
            else:
                print("none")
                break
        return cnt

if __name__ == '__main__':
    cam = Camera()
    cam_num = cam.get_cam_num()

