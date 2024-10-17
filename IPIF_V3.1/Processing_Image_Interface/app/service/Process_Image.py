import os
from multiprocessing import *
from fastapi.responses import JSONResponse
from Processing_Image_Interface.app.model.Data_Info_Model import *
from Processing_Image_Interface.app.model.Unprocess_Entity_Model import *

from Processing_Image_Interface.utilities.Remove_Background_Image import *

from Processing_Image_Interface.logger.Logger import *

class Image_Config(object):
    img_size_inp: str
    omp_num_threads: str
    data_in: str
    data_out: str

def is_image_file(file_name):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    return any(file_name.lower().endswith(ext) for ext in image_extensions)

def processImage(data_if: Data_Info_Model, num_processes: int):
    _logger = Logger(folder_name="log_Image_Processing_IF", file_name="MainProcess", name_logger="MainProcess", file_mode="a")
    if Image_Config.omp_num_threads != "INF":
        os.environ["OMP_NUM_THREADS"] = Image_Config.omp_num_threads
    img_size_inp = Image_Config.img_size_inp
    data_if_rtn = Data_Info_Rtn_Model(data_in=data_if.data_in, data_out=data_if.data_out)
    try:
        # data_path = data_if.data_in
        # data_out_path = data_if.data_out

        ### ADJUST
        data_path = Image_Config.data_in
        data_out_path = Image_Config.data_out

        _logger.trace(f"Data In: {data_path}")
        _logger.trace(f"Data Out: {data_out_path}")
        all_files = os.listdir(data_path)
        list_image = [file for file in all_files if is_image_file(file)]
        total_data = len(list_image)
        _logger.trace(f"Total Data: {total_data}")

        if(total_data < num_processes):
            _logger.trace("Total image less than num process. Use main process for processing")
            process_remove_background_image(data_path=data_path, img_size_inp=img_size_inp, lImage=list_image, data_out_path=data_out_path, name_logger_sub_process="MainProcess")
            data_if_rtn.status = "Process data successful"
            _logger.trace("Process data successful")
            return JSONResponse(content=data_if_rtn.model_dump(mode="json"), status_code=200)
        else:
            chunk_size = total_data // num_processes
            chunks = [list_image[i:i + chunk_size] for i in range(0, len(list_image), chunk_size)]

            if len(chunks) > num_processes:
                chunks[-2].extend(chunks[-1])   
                chunks = chunks[:-1]  

            processes = []

            for i, chunk in enumerate(chunks):
                print(f"Chunk {i + 1}: {chunk[:10]}...")
                print(f"Len chunk: {len(chunk)}")
                p: Process
                name_logger_sub_process = f"Subprocess_{str(i)}"
                p = Process(target=process_remove_background_image, args=(data_path, img_size_inp, chunk, data_out_path, name_logger_sub_process))
                p.start()
                processes.append(p)
            for process in processes:
                process.join()

            data_if_rtn.status = "Process data successful"
            _logger.trace("Process data successful")
            return JSONResponse(content=data_if_rtn.model_dump(mode="json"), status_code=200)
    except Exception as e:
        data_if_rtn.status = str(e)
        _logger.trace(f"Error: {str(e)}")
        return JSONResponse(content=data_if_rtn.model_dump(mode="json"), status_code=400)

def load_img_config(img_size_inp: str, omp_num_threads: int, data_in: str, data_out: str):
    Image_Config.img_size_inp = img_size_inp
    Image_Config.omp_num_threads = str(omp_num_threads)
    Image_Config.data_in = data_in
    Image_Config.data_out = data_out