import torch

import torch.multiprocessing as mp

print("Is cuda availible? ", torch.cuda.is_available())

def test(match_id):
    print("Test no: ", match_id)
    return match_id

if __name__ == "__main__":
    
    mp.set_start_method('spawn', force=True)

    result_list = []
    match_id = 1
    args_list = [(match_id,) for match_id in range(1, 6)]
    
    print("wazzup")
    with mp.Pool(mp.cpu_count()) as pool:
        result_list = pool.starmap(test, args_list)
        
    print(result_list)
    
    