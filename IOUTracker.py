# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# Modified by GBJim
# Modified by Rudy Nurhadi
# ---------------------------------------------------------

class IOUTracker():
    def __init__(self, sigma_l=0, sigma_h=0.5, sigma_iou=0.5, t_max=5, verbose=False):
        self.sigma_l = sigma_l
        self.sigma_h = sigma_h
        self.sigma_iou = sigma_iou
        self.t_max = t_max
        self.frame = 0
        self.id_count = 0
        self.tracks_active = {}
        self.verbose = verbose

    #Clear the old tracks
    def clean_old_tracks(self):
        target_frame = self.frame - self.t_max
        if target_frame in self.tracks_active:
            if self.verbose:
                print("[LOG]: Tracks Deleted: {}".format(self.tracks_active[target_frame].keys()))
            del(self.tracks_active[target_frame])

    #Retrieve tracks in an correct matching order
    def retrieve_tracks(self):
        tracks = []
        frames = range(self.frame, self.frame - self.t_max, -1)
        for frame in frames: 
            if frame in self.tracks_active:
                if self.verbose:
                    print("[LOG]: Frame {} Tracks Retrieved: {}".format(frame,self.tracks_active[frame].keys()))
                tracks += self.tracks_active[frame].items()
        return tracks

    #Get all active tracks
    def get_active_tracks(self):
        tracks = {}
        frames = range(self.frame - self.t_max - 1, self.frame + 1)
        for frame in frames:
            if frame in self.tracks_active:
                if self.verbose:
                    print("[LOG]: Frame {} Tracks Retrieved: {}".format(frame,self.tracks_active[frame].keys()))
                for _id in self.tracks_active[frame].keys():
                    tracks[_id] = self.tracks_active[frame][_id]
                    if frame == self.frame:
                        tracks[_id]['active'] = True
                    else:
                        tracks[_id]['active'] = False
        return tracks

    def track(self, detections):
        self.frame += 1
        self.tracks_active[self.frame] = {}
        #Clear the tracks in old frame
        self.clean_old_tracks()

        dets = [det for det in detections if det['score'] >= self.sigma_l]
        
        for id_, track in self.retrieve_tracks():
            if len(dets) > 0:
                # get det with highest iou
                best_match = max(dets, key=lambda x: iou(track['bbox'], x['bbox']))
                if iou(track['bbox'], best_match['bbox']) >= self.sigma_iou:
                    best_match['start_point'] = track['start_point']
                    self.tracks_active[self.frame][id_] = best_match
                    if self.verbose:
                        print("[LOG]: Tracke {} updated with bbox: {}".format(id_ , best_match['bbox']))
                    # remove from best matching detection from detections
                    del dets[dets.index(best_match)]

        #Create new tracks
        for det in dets:
            self.id_count += 1
            xcenter = int((det['bbox'][0] + det['bbox'][2]) / 2)
            ycenter = int((det['bbox'][1] + det['bbox'][3]) / 2)
            det['start_point'] = (xcenter, ycenter)
            self.tracks_active[self.frame][self.id_count] = det
            if self.verbose:
                print("[LOG]: Tracke {} added with bbox: {}".format(self.id_count, det['bbox']))

        #Return the current tracks 
        if self.verbose:
            print("[LOG]: Trackes {} returned".format(self.tracks_active[self.frame].keys()))
        return self.tracks_active[self.frame]

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / (size_union + 1e-05)
