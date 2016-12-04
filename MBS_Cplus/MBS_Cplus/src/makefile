INCLUDE=-I/usr/local/include -I/usr/local/include/opencv -I/usr/local/include/opencv2
LIBS_opencv=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lm -lopencv_videostab -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_legacy -lpthread -lrt
main:main.o MBS.o
	g++ main.o MBS.o -o main ${LIBS_opencv} ${INCLUDE}
main.o:main.cpp MBS.hpp
	g++ -c main.cpp
MBS.o:MBS.cpp MBS.hpp
	g++ -c MBS.cpp
clean:
	rm -f *.o
	rm -f main
