import face_recognition
import cv2
import datetime

f=open("Date_time.txt","a+")
video=cv2.VideoCapture(0) #using webcam to load a video
sahil=face_recognition.load_image_file("/home/sahil/face.jpg") #loading image database 
sahil_encoding=face_recognition.face_encodings(sahil)[0]  #Extracting required features from image

nikhil=face_recognition.load_image_file("/home/sahil/nikhil.jpg") #loading image database 
nikhil_encoding=face_recognition.face_encodings(nikhil)[0] 
face_encoding=[ sahil_encoding, nikhil_encoding]  #loading face encoding to array
face_name=["Sahil","Nikhil"]              #loading

#initialising variables for video frames
new_encodings=[]
new_locations=[]
new_names=[]
match=True
flag=0

while(1):
	ret,frame=video.read()
	rgb_frame=frame[:, :, ::-1]  #converts BGR image of opencv to RGB required for face_recognition
	if match:
		new_locations=face_recognition.face_locations(rgb_frame)
		new_encodings=face_recognition.face_encodings(rgb_frame,new_locations)
		new_names=[]
		name= "unknown"
		for new_encoding in new_encodings:
			matches=face_recognition.compare_faces(face_encoding,new_encoding,tolerance=0.45)
			if True in matches:
				first_match_index=matches.index(True)
				name=face_name[first_match_index]		
				flag=1
			else:
				flag=2
			new_names.append(name)
	match= not match
	for (top, right, bottom, left),name in zip(new_locations, new_names):
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)		
	cv2.imshow('Video', frame)
	if flag==1:
		date_string = datetime.datetime.now().strftime("%d-%m-%y-%H:%M")
		f.write(name)
		f.write("\t")
		f.write(date_string)
		f.write("\n")	
		break
	
	if flag==2:
		date_string = datetime.datetime.now().strftime("%d-%m-%y-%H:%M")
		f.write(name)
		f.write("\t")
		f.write(date_string)
		f.write("\n")
		break	
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

video.release()
cv2.destroyAllWindows()

print(new_names)			
f.close()	
		



