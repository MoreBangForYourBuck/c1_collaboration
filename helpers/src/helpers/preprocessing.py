from typing import Dict


def expand_ann(imu_t:list, ann:list, ann_t:list) -> Dict[str, list]:
    class MemIter():
        def __init__(self, v:list):
            self.v = v
            self.i = 0
            self.val = v[self.i]
            
        def increment(self):
            if self.i < len(self.v) - 1: # If not at end of list
                self.i += 1
                self.val = self.v[self.i]


    mem_ann = MemIter(ann)
    mem_ann_t = MemIter(ann_t)

    ann_out = []
    for i, curr_imu_time in enumerate(imu_t):
        
        if curr_imu_time < mem_ann_t.val:
            ann_out.append(mem_ann.val)
            
        else:
            ann_out.append(mem_ann.val)
            mem_ann.increment()
            mem_ann_t.increment()
                
    return {
        'ann': ann_out,
        'ann_time': imu_t
    }
