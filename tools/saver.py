import json

class Saver:

    def __init__(self,path='sample.json'):
        self.path = path
        self.tracks = {}

    def update(self,data,extra_keys=None,extra_vals=None):

        if data['id'] not in self.tracks:
            self.tracks[data['id']] = {
                'label': data['label'],
                'timestamps': [],
                'x':[],
                'y':[],
                'radius':[]}
            if extra_keys is not None:
                for i in extra_keys:
                    self.tracks[data['id']][i] = []
                
        x_mid = data['x']
        y_mid = data['y']
        timestamp = data['timestamp']
        radius = data['radius']

        self.tracks[data['id']]['timestamps'].append(timestamp)
        self.tracks[data['id']]['x'].append(x_mid)
        self.tracks[data['id']]['y'].append(y_mid)
        if radius != -1:
            self.tracks[data['id']]['radius'].append(radius)
        if extra_keys is not None:
            for i,j in zip(extra_keys,extra_vals):
                self.tracks[data['id']][i].append(j)
    
    def add_key(self,key_name):
        for i in self.tracks.keys():
            if key_name not in list(self.tracks[i].keys()):
                self.tracks[i][key_name] = []
        
    def save(self,keys_dropped=None):
        if keys_dropped != None:
            for k in keys_dropped:
                res = self.tracks.pop(k,None)
                if res != None:
                    print('dropped: {}'.format(k))
        with open(self.path,'w') as f:
            json.dump(self.tracks,f)