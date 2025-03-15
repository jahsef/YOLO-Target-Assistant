

class PevTracker:
    def __init__(self):
        self.uid = 0
        
        self.detections = {
            # 'uid': {
            #     'x1': None,#int
            #     'y1': None,#int
            #     'x2': None,#int
            #     'y2': None,#int
            #     'motion_vector':None,#tuple[int,int]
            #     'last_detection':None#int
            # }
        }

    def update_detections(self,frame_bounding_boxes : list[tuple[int,int,int,int]]):
        #if nothing in dict, just assign everything a uid?
        #should probably have use an algorithm for associating bounding boxes with uids
        # if len(self.detections) == 0:
        #     for bounding_box in frame_bounding_boxes:
        #         x1,y1,x2,y2 = bounding_box
        #         self._create_new_uid(x1,y1,x2,y2)
            
        
        
        self._associate_detections(frame_bounding_boxes)
        
        self._iterate_last_detection()
        
    def _associate_detections(self,frame_bounding_boxes):
        #tries to map a bounding box to a uid
        #maybe the center and h/w is within a certain threshold?
        #within 20 pixels center and within 20 pixels h/w?
        for bounding_box in frame_bounding_boxes:
            # n^2 complexity probably not good but prolly have max like 5 detections
            x1,y1,x2,y2 = bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
            
            new_center = (x2 - x1, y2 - y1)
            
            keys = list(self.detections.keys())
            has_been_associated = False

            for uid in keys:

                if self.detections[uid]['last_detection'] == 0:#skip associating to new uid
                    
                    continue
                curr_uid_bb = (self.detections[uid]['x1'],self.detections[uid]['y1'],self.detections[uid]['x2'],self.detections[uid]['y2'])
                curr_uid_center = (curr_uid_bb[2] - curr_uid_bb[0], curr_uid_bb[3] - curr_uid_bb[1])
                
                delta_x = curr_uid_center[0] - new_center[0]
                delta_y = curr_uid_center[1] - new_center[1]
                dist_sq = delta_x **2 + delta_y**2

                if dist_sq <= 400:#if under threshold then associate with that uid and then break loop
                    #should probably make motion vector calc a bit better (over multipe frames)
                    self._update_uid(uid,x1,y1,x2,y2, delta_x, delta_y)
                    # print('associating')
                    has_been_associated = True
                    break
            if not has_been_associated:#if cant be associated with a uid, make a new one
                # print('cant be associated creating new uid')
                self._create_new_uid(x1,y1,x2,y2)
                
            
            
                    
                
                

        pass
    def _create_new_uid(self,x1,y1,x2,y2):
        self.detections[self.uid] = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'motion_vector': None,#should i do none or 0,0?
            'last_detection': 0
        }
        self.uid +=1
    
    def _update_uid(self, uid, x1,y1,x2,y2, delta_x, delta_y ):
        self.detections[uid]['x1'],self.detections[uid]['y1'],self.detections[uid]['x2'],self.detections[uid]['y2'] = x1,y1,x2,y2
        self.detections[uid]['motion_vector'] = (delta_x,delta_y)
        self.detections[uid]['last_detection'] = 0

    def _iterate_last_detection(self):
        keys = list(self.detections.keys())
        for uid in keys:
            if self.detections[uid]['last_detection'] >=5:
                del self.detections[uid]
            else:
                self.detections[uid]['last_detection'] +=1

                
        
          

    
if __name__ == "__main__":
    import numpy as np
    import time
    
    tracker = PevTracker()
    print('starting')
    start = time.perf_counter()
    
    num_detections = 16
    num_frames = 1440
    # for i in range(num_frames):
    #     poople = []
    #     for j in range(num_detections):
    #         x1,y1,x2,y2 = np.random.randint(0,1440),np.random.randint(0,1440),np.random.randint(0,1440),np.random.randint(0,1440)
    #         if x1 > x2:
    #             x1,x2 = x2,x1
    #         if y1 > y2:
    #             y1,y2 = y2,y1
    #         poople.append((x1,y1,x2,y2))
    #     tracker.update_detections(poople)
    # print(f'time: {(time.perf_counter() - start):.4f}')
    # print(len(tracker.detections))
    
    new_detections = [(1,2,3,4)]
    tracker.update_detections(new_detections)
    print(tracker.detections)
    new_detections = [(2,3,3,5)]
    tracker.update_detections(new_detections)
    print(tracker.detections)
    