from QuadScanTasks import *
import time

if __name__ == "__main__":
    test = "process_images"
    if test == "process_pool":
        t1 = ProcessPoolTask(test_f, name="process_test")
        t1.start()
        t1.add_work_item(0.5)
        t1.add_work_item("wow")
        # time.sleep(2.0)
        t1.finish_processing()
        print("Completed: {0}".format(t1.completed))
        print("Final result 1: {0}".format(t1.get_result(wait=True)))
    elif test == "process_image":
        t1 = ImageProcessorTask(threshold=0.0, cal=[1.0, 1.0], kernel=3, name="image_processor")
        t1.start()
        image_name = "03_03_1.035_.png"
        path_name = "..\\..\\emittancesinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        # path_name = "D:\\Programmering\emittancescansinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        t2 = LoadQuadImageTask(image_name, path_name, name="load_im", callback_list=[t1.process_image])
        # t2.add_callback(t1.process_image)
        t2.start()
        # quad_image = t2.get_result(True)
        #
        # t1.process_image(quad_image)
        print("Got image. {0}".format(t1.get_result(True)))
        t1.stop_processing()
    elif test == "process_images":
        # t1 = ImageProcessorTask(threshold=0.0, cal=[1.0, 1.0], kernel=3, name="image_processor")
        # t1.start()
        path_name = "..\\..\\emittancesinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        # path_name = "D:\\Programmering\emittancescansinglequad\\saved-images\\2018-04-16_13-40-48_I-MS1-MAG-QB-01_I-MS1-DIA-SCRN-01"
        t2 = LoadQuadScanDirTask(path_name, name="quad_dir")
        t2.start()
        # quad_image = t2.get_result(True)
        #
        # t1.process_image(quad_image)
        print("Got image. {0}".format(t2.get_result(True)))


