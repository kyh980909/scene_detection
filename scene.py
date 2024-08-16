import cv2
import numpy as np
import math

def get_psnr(I1, I2):
    s1 = cv2.absdiff(I1, I2)        # |I1 - I2|
    s1 = s1.astype(np.float32)      # cannot make a square on 8 bits
    s1 = s1 ** 2                    # |I1 - I2|^2
    s = np.sum(s1, axis=(0, 1))     # sum elements per channel
    sse = np.sum(s)                 # sum channels
    
    if sse <= 1e-10:  # for small values return zero
        return 0
    
    mse = sse / float(I1.shape[0] * I1.shape[1] * I1.shape[2])
    psnr = 10.0 * math.log10((255 * 255) / mse)
    return psnr

def merge(m1, m2, result):
    result = cv2.resize(result, (m1.shape[1] + m2.shape[1], m1.shape[0]))

    result[0:m1.shape[0], 0:m1.shape[1]] = m1
    result[0:m2.shape[0], m1.shape[1]:] = m2

    cv2.putText(result, "Normal Video", (30, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (200, 200, 250), 1, cv2.LINE_AA)

    cv2.putText(result, "Scene Change Detection", (m1.shape[1] + 30, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (200, 200, 250), 1, cv2.LINE_AA)
    return result

def main():
    frame_num = -1  # Frame counter
    psnr_v = 0
    CHANGE_DETECT_RATIO = 15.0
    video_path = "video/lion.mp4"
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video - {video_path}")
        return -1
    
    s = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    prev_frame = None
    curr_frame = None
    change_frame = None
    result = np.zeros((s[1], s[0] * 2, 3), dtype=np.uint8)
    
    cv2.namedWindow("Scene Change Detection")
    cv2.resizeWindow("Scene Change Detection", s[0] * 2, s[1])
    
    while True:
        frame_num += 1
        ret, curr_frame = cap.read()
        
        if not ret:
            print(f"End of video or cannot read the frame at frame {frame_num}")
            break
        
        if curr_frame is None or curr_frame.size == 0:
            print(f"Empty frame at frame {frame_num}")
            break
        
        if frame_num < 1:
            prev_frame = curr_frame.copy()
            change_frame = curr_frame.copy()
            continue
        
        psnr_v = get_psnr(prev_frame, curr_frame)
        print(psnr_v)
        if psnr_v < CHANGE_DETECT_RATIO:
            change_frame = curr_frame.copy()
        result = merge(curr_frame, change_frame, result)
        cv2.imshow("Scene Change Detection", result)
        
        if frame_num % 2 == 0:
            prev_frame = curr_frame.copy()
        
        if cv2.waitKey(10) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()