import cv2


def annotate_video_per_frame(input_path, output_path, ann_arrays, format_string='{}', max_count=None):
    cap = cv2.VideoCapture(input_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    wrt = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (frame_width, frame_height))

    no_break = True
    count = 0
    while no_break:
        no_break, frame = cap.read()

        for num, ann_arr in enumerate(ann_arrays):
            cv2.putText(
                frame, format_string.format(ann_arr[count]),
                (60 + num * 60, 60), cv2.FONT_HERSHEY_DUPLEX, 3., (255, 255, 0)
            )
        wrt.write(frame)
        count += 1
        if max_count and count == max_count:
            no_break = False

    wrt.release()
    cap.release()
