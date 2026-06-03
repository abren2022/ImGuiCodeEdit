    def _copy_id_new_object(self,bbox: List[float], reid:int, obj_type: str):
        new_id = reid
        if new_id in self.active_objects and self.active_objects[new_id]['type']==obj_type:
            self.active_objects[new_id]['last_bbox']=bbox
            self.active_objects[new_id]['last_seen_frame'] = self.current_frame
        else:
            self.active_objects[new_id] = {
                'last_bbox': bbox,
                'last_seen_frame': self.current_frame,
                'type': obj_type
            }
        return new_id
