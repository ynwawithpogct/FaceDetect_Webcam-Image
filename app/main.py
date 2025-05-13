import numpy as np
import cv2
import insightface
from datetime import datetime
import os
import json
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import FileResponse
import logging
from pydantic import BaseModel
import unicodedata

class Person:
    def __init__(self, name: str, embedding: np.ndarray, created_time: str = '', updated_time: str = ''):
        if not isinstance(name, str):
            raise ValueError(f"Expect value name is String but got {type(name)}")
        self.name = name
        if embedding.ndim != 1:
            raise ValueError(f"Expect embedding is 1-dimensional array but got {embedding.ndim}-dimensional array")
        self.embedding = embedding
        # Setup datetime
        if created_time == '' and updated_time == '':
            self.created_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            self.updated_time = self.created_time
        elif created_time == '':
            raise ValueError("Got updated_time but created_time is not given")
        elif updated_time == '':
            raise ValueError("Got created_time but updated_time is not given")
        elif not (isinstance(created_time, str) and self.is_valid_datetime_format(created_time)):
            raise ValueError(f"Expect created_time is String with format '%d-%m-%Y %H:%M:%S' but got {created_time}")
        elif not (isinstance(updated_time, str) and self.is_valid_datetime_format(updated_time)):
            raise ValueError(f"Expect updated_time is String with format '%d-%m-%Y %H:%M:%S' but got {updated_time}")
        else:
            self.created_time = created_time
            self.updated_time = updated_time
            
    @classmethod
    def load(cls, embedding_path: str, dict_path: str):
        if not os.path.exists(embedding_path):
            raise ValueError(f" embedding_path: {embedding_path} does not exist")
        embedding = np.load(embedding_path)
        if not os.path.exists(dict_path):
            raise ValueError(f" dict_path: {dict_path} does not exist")
        with open(dict_path, 'r') as f:
            person_dict = json.load(f)
        return cls(person_dict['name'], embedding, person_dict['created_time'], person_dict['updated_time'])
        
    def update(self, embedding: np.ndarray):
        if embedding.ndim != 1:
            raise ValueError(f"Expect embedding is 1-dimensional array but got {embedding.ndim}-dimensional array")
        self.embedding = embedding
        # Update datetime
        self.updated_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
    def update_name(self, name:str):
        if not isinstance(name, str):
            raise ValueError(f"Expect value name is String but got {type(name)}")
        self.name = name
        # Update datetime
        self.updated_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
    def update_all(self, name:str, embedding: np.ndarray):
        if not isinstance(name, str):
            raise ValueError(f"Expect value name is String but got {type(name)}")
        self.name = name
        if embedding.ndim != 1:
            raise ValueError(f"Expect embedding is 1-dimensional array but got {embedding.ndim}-dimensional array")
        self.embedding = embedding
        # Update datetime
        self.updated_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
    def distance(self, embedding: np.ndarray, metric: str='cosine') -> float:
        if embedding.ndim != 1:
            raise ValueError(f"Expect embedding is 1-dimensional array but got {embedding.ndim}-dimensional array")
        if self.embedding.shape[0] != embedding.shape[0]:
            raise ValueError(f"Give embedding have diffence shape to compare with Person embedding: {embedding.shape[0]} & {self.embedding.shape[0]}")
        # Calculate distance
        if metric == 'cosine':
            return 1 - (np.dot(self.embedding, embedding) / (np.linalg.norm(self.embedding) * np.linalg.norm(embedding)))
        elif metric == 'euclidean':
            return np.linalg.norm(self.embedding - embedding)
        elif metric == 'manhattan':
            return np.sum(np.abs(self.embedding - embedding))
        else:
            raise ValueError(f"Invalid metric. Expect metric in [cosine, euclidean, manhattan] but of {metric}.")
        
    def save(self, embedding_path: str, dict_path: str):
        np.save(embedding_path, self.embedding)
        person_dict = {
            'name': self.name,
            'created_time': self.created_time,
            'updated_time': self.updated_time
        }
        with open(dict_path, 'w') as f:
            json.dump(person_dict, f)
            
    def get(self) -> dict:
        return {
            'name': self.name,
            'created_time': self.created_time,
            'updated_time': self.updated_time
        }
        
    def get_name(self) -> str:
        return self.name
        
    @staticmethod
    def is_valid_datetime_format(s: str) -> bool:
        # check datetime format
        try:
            datetime.strptime(s, "%d-%m-%Y %H:%M:%S")
            return True
        except ValueError:
            return False
        
class Embeddings_Store:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        id_path = os.path.join(path, 'id_dict.json')
        if not os.path.exists(id_path):
            self.id_dict = {}
            with open(id_path, "w") as f:
                json.dump(self.id_dict, f)
        else:
            with open(id_path, 'r') as f:
                self.id_dict = json.load(f)
                if not isinstance(self.id_dict, dict):
                    raise ValueError(f"Expect id_dict loaded form id_dict.json is dictionary but got {type(self.id_dict)}")
        self.embedding_dict = {key: Person.load(value['embedding_path'], value['dict_path']) for key, value in self.id_dict.items()}
    
    def add(self, name: str, embedding: np.ndarray) -> str:
        if isinstance(name, str) and isinstance(embedding, np.ndarray):
            new_id = self.create_id()
            self.id_dict[new_id] = {
                'embedding_path': os.path.join(self.path, new_id+'.npy'),
                'dict_path': os.path.join(self.path, new_id+'.json')
            }
            self.embedding_dict[new_id] = Person(name, embedding)
            self.embedding_dict[new_id].save(self.id_dict[new_id]['embedding_path'], self.id_dict[new_id]['dict_path'])
        else:
            raise ValueError(f"Expect (name_list, embedding_list) are (str, np.ndarray) but got ({type(name)}, {type(embedding)})")
        id_path = os.path.join(self.path, 'id_dict.json')
        with open(id_path, "w") as f:
            json.dump(self.id_dict, f)
        return new_id
           
    def update(self, id:str, name: str|None=None, embedding: np.ndarray|None=None):
        if name is not None and embedding is not None:
            self.embedding_dict[id].update_all(name, embedding)
        elif name is not None:
            self.embedding_dict[id].update_name(name)
        elif embedding is not None:
            self.embedding_dict[id].update(embedding)
        if name is not None or embedding is not None:
            self.embedding_dict[id].save(self.id_dict[id]['embedding_path'], self.id_dict[id]['dict_path'])
            
    def delete(self, id) -> bool:
        if id not in self.id_dict:
            if id in self.embedding_dict:
                raise (f"Data error found {id} in embedding_dict but not in id_dict")
            else:
                return False
        else:
            if id in self.embedding_dict:
                del self.embedding_dict[id]
                os.remove(self.id_dict[id]['embedding_path'])
                os.remove(self.id_dict[id]['dict_path'])
                del self.id_dict[id]
                id_path = os.path.join(self.path, 'id_dict.json')
                with open(id_path, 'r') as f:
                    self.id_dict = json.load(f)
                return True
            else:
                raise (f"Data error found {id} in id_dict but not in embedding_dict")
            
    def find_embedding(self, embedding: np.ndarray, metric: str='cosine', threshold: float=0.5, return_id: bool = False):
        if isinstance(embedding, np.ndarray):
            distance_dict = {key: value.distance(embedding, metric) for key, value in self.embedding_dict.items()}
            best_id = min(distance_dict, key=distance_dict.get)
            if distance_dict[best_id] < threshold:
                return best_id if return_id else self.embedding_dict[best_id].get_name()
            else:
                return None
        else:
            raise ValueError(f"Expect embedding_list is np.ndarray but got {type(embedding)}")
        
    def get_data(self) -> list[dict]:
        return {key: value.get() for key, value in self.embedding_dict.items()}
    
    def get_person(self, id) -> Person:
        if id not in self.id_dict:
            if id in self.embedding_dict:
                raise (f"Data error found {id} in embedding_dict but not in id_dict")
            else:
                return None
        else:
            if id in self.embedding_dict:
                return self.embedding_dict[id]
            else:
                raise (f"Data error found {id} in id_dict but not in embedding_dict")
        
    def create_id(self) -> str:
        ids = [key for key in self.id_dict]
        while True:
            new_id = uuid.uuid4()
            if new_id not in ids:
                return str(new_id)
            
class History:
    def __init__(self, path, image_path=None):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.image_path = os.path.join(path, 'image') if image_path is None else image_path
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        id_path = os.path.join(path, 'id_dict.json')
        if not os.path.exists(id_path):
            self.id_dict = {}
            with open(id_path, "w") as f:
                json.dump(self.id_dict, f)
        else:
            with open(id_path, 'r') as f:
                self.id_dict = json.load(f)
                if not isinstance(self.id_dict, dict):
                    raise ValueError(f"Expect id_dict loaded form id_dict.json is dictionary but got {type(self.id_dict)}")
            
    def add(self, person_id, person: Person, bbox:np.ndarray, image) -> str:
        new_id = self.create_id()
        self.id_dict[new_id] = {
            'type': 'Add',
            'person': True,
            'person_id': person_id,
            'embedding_path': os.path.join(self.path, new_id+'.npy'),
            'dict_path': os.path.join(self.path, new_id+'.json'),
            'image': True,
            'name': [person.get()['name']],
            'bbox_path': os.path.join(self.image_path, new_id+'.npz'),
            'image_path': os.path.join(self.image_path, new_id+'.jpg'),
            'time': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = unicodedata.normalize("NFD", self.id_dict[new_id]['name'][0])
        text_no_diacritics = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        cv2.putText(image, text_no_diacritics, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(self.id_dict[new_id]['image_path'], image)
        np.savez(self.id_dict[new_id]['bbox_path'], **{f"person_{self.id_dict[new_id]['name'][i]}": box for i, box in enumerate([bbox])})
        person.save(self.id_dict[new_id]['embedding_path'], self.id_dict[new_id]['dict_path'])
        id_path = os.path.join(self.path, 'id_dict.json')
        with open(id_path, "w") as f:
            json.dump(self.id_dict, f)
        return self.id_dict[new_id]['image_path']
        
    def delete(self, person_id, person: Person):
        new_id = self.create_id()
        self.id_dict[new_id] = {
            'type': 'Delete',
            'person': True,
            'person_id': person_id,
            'embedding_path': os.path.join(self.path, new_id+'.npy'),
            'dict_path': os.path.join(self.path, new_id+'.json'),
            'image': False,
            'time': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }
        person.save(self.id_dict[new_id]['embedding_path'], self.id_dict[new_id]['dict_path'])
        id_path = os.path.join(self.path, 'id_dict.json')
        with open(id_path, "w") as f:
            json.dump(self.id_dict, f)
            
    def update(self, person_id, person: Person, bbox:np.ndarray|None=None, image=None) -> str|None:
        new_id = self.create_id()
        if bbox is not None:
            self.id_dict[new_id] = {
                'type': 'Update',
                'person': True,
                'person_id': person_id,
                'embedding_path': os.path.join(self.path, new_id+'.npy'),
                'dict_path': os.path.join(self.path, new_id+'.json'),
                'image': True,
                'name': [person.get()['name']],
                'bbox_path': os.path.join(self.image_path, new_id+'.npz'),
                'image_path': os.path.join(self.image_path, new_id+'.jpg'),
                'time': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            }
        else:
            self.id_dict[new_id] = {
                'type': 'Update',
                'person': True,
                'person_id': person_id,
                'embedding_path': os.path.join(self.path, new_id+'.npy'),
                'dict_path': os.path.join(self.path, new_id+'.json'),
                'image': False,
                'time': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            }
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = unicodedata.normalize("NFD", self.id_dict[new_id]['name'][0])
            text_no_diacritics = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            cv2.putText(image, text_no_diacritics, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(self.id_dict[new_id]['image_path'], image)
            np.savez(self.id_dict[new_id]['bbox_path'], **{f"person_{self.id_dict[new_id]['name'][i]}": box for i, box in enumerate([bbox])})
        person.save(self.id_dict[new_id]['embedding_path'], self.id_dict[new_id]['dict_path'])
        id_path = os.path.join(self.path, 'id_dict.json')
        with open(id_path, "w") as f:
            json.dump(self.id_dict, f)
        return self.id_dict[new_id]['image_path'] if bbox is not None else None
            
    def detect(self, name: list[str], bbox: list[np.ndarray], image) -> str: #, distance: list
        new_id = self.create_id()
        self.id_dict[new_id] = {
            'type': 'Update',
            'person': False,
            'image': True,
            'name': name,
            'bbox_path': os.path.join(self.image_path, new_id+'.npz'),
            'image_path': os.path.join(self.image_path, new_id+'.jpg'),
            'time': datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }
        for bbox, label in zip(bbox, name):
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = unicodedata.normalize("NFD", label)
            text_no_diacritics = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            cv2.putText(image, text_no_diacritics, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(self.id_dict[new_id]['image_path'], image)
        np.savez(self.id_dict[new_id]['bbox_path'], **{f"person_{label}": box for label, box in zip(bbox, name)})
        id_path = os.path.join(self.path, 'id_dict.json')
        with open(id_path, "w") as f:
            json.dump(self.id_dict, f)
        return self.id_dict[new_id]['image_path']
        
    def create_id(self):
        ids = [key for key in self.id_dict]
        while True:
            new_id = uuid.uuid4()
            if new_id not in ids:
                return str(new_id)
            
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataframe")
data = Embeddings_Store(data_path)

history_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history")
image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "image")
history = History(history_path, image_path)

face_rec_model = insightface.app.FaceAnalysis(name="buffalo_l")  # ResNet100, ArcFace, Glint360K
face_rec_model.prepare(ctx_id=0)

threshold = 0.5
metric = 'cosine'
return_id = False

logging.basicConfig(level=logging.INFO)


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

@app.get("/data/get_data")
def get_data():
    global data
    try:
        return {"Data": data.get_data()}
    except Exception as e:
        logging.error("Error: %s", str(e))
        raise HTTPException(status_code=400, detail="Can not get data")
    
@app.delete("/data/delete_data/")
def delete_data(item_id: str = Body(...)):
    global data, history
    try:
        temp_person = data.get_person(item_id)
        if temp_person is None:
            raise HTTPException(status_code=402, detail="Can not found person")
        else:
            check = data.delete(item_id)
            history.delete(item_id, temp_person)
        return {"Item_id": item_id, "Deleted": check}
    except Exception as e:
        logging.error("Error: %s", str(e))
        raise HTTPException(status_code=400, detail="Data Fail")
    
@app.put("/data/add_data")
async def add_data(name: str = Body(...), file: UploadFile = File(...)):
    global data, history, face_rec_model
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=404, detail="file is not image")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = face_rec_model.get(img)
        embedding = faces[0].embedding  # vector 512 chiều
        item_id = data.add(name, embedding)
        temp_person = data.get_person(item_id)
        if temp_person is None:
            raise HTTPException(status_code=402, detail="Can not found person")
        else:
            history.add(item_id, temp_person, faces[0].bbox, img)
        return {"Item_id": item_id, "Added": True}
    except Exception as e:
        logging.error("Error: %s", str(e))
        raise HTTPException(status_code=400, detail="Can not add data")
    

class UpdateRequest(BaseModel):
    item_id: str
    name: str
    
@app.patch("/data/update_data/update_name/")
def update_name(req: UpdateRequest):
    global data, history
    item_id = req.item_id
    name = req.name
    try:
        temp_person = data.get_person(item_id)
        if temp_person is None:
            raise HTTPException(status_code=402, detail="Can not found person")
        else:
            data.update(item_id, name=name)
            history.update(item_id, temp_person)
        return {"Item_id": item_id, "Updated": True}
    except Exception as e:
        logging.error("Error: %s", str(e))
        raise HTTPException(status_code=400, detail="Can not update data name")
    
@app.patch("/data/update_data/update_embedding/")
async def update_embedding(item_id: str = Body(...), file: UploadFile = File(...)):
    global data, history, face_rec_model
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=404, detail="file is not image")
    try:
        temp_person = data.get_person(item_id)
        if temp_person is None:
            raise HTTPException(status_code=402, detail="Can not found person")
        else:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = face_rec_model.get(img)
            embedding = faces[0].embedding  # vector 512 chiều
            data.update(item_id, embedding=embedding)
            history.update(item_id, temp_person, faces[0].bbox, img)
        return {"Item_id": item_id, "Updated": True}
    except Exception as e:
        logging.error("Error: %s", str(e))
        raise HTTPException(status_code=400, detail="Can not update data name")
    
@app.post("/model/detect")
async def detect(file: UploadFile = File(...)):
    global face_rec_model, data, history, metric, threshold, return_id
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=404, detail="file is not image")
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = face_rec_model.get(img)
        detected_list = []
        bbox_list = []
        for face in faces:
            detected_face = data.find_embedding(face.embedding, metric, threshold, return_id)
            if detected_face is not None:
                detected_list.append(detected_face)
                bbox_list.append(face.bbox)
        image_path = history.detect(detected_list, bbox_list, img)
        return FileResponse(image_path, media_type="image/jpeg")
        
    except Exception as e:
        logging.error("Error: %s", str(e))
        raise HTTPException(status_code=400, detail="Can not detect object")