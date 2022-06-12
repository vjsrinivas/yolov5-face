from loguru import logger
import cupy as cp

def tic():
    t1 = cp.cuda.Event()
    t1.record()
    return t1

def toc(t1, prefix=""):
    t2 = cp.cuda.Event()
    t2.record()
    t2.synchronize()
    _time = cp.cuda.get_elapsed_time(t1,t2)/1000.0
    #print( "%s: %f"%(prefix, _time))
    logger.info("%s: %f"%(prefix, _time))
    return _time

