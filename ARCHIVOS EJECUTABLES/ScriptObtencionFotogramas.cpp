#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <unordered_set>

using namespace std;
using namespace cv;

vector<string> videoFiles(string);
vector<string> getWords(string, string);
void saveFrames(const vector<Mat>&, const vector<int>&, const vector<vector<string>>&, const string);
vector<double> getIntervals(string, string);
vector<string> getGroups(string);
void createDirectories(const string, const vector<vector<string>>&);
void getFrames(const string, vector<Mat>&, vector<int>&, const vector<double>&);
bool createDirectory(const string&);
void startProcess(const string&);

int main() {

	std::cout << "Inicio del programa" << std::endl;
	std::cout << "Leyendo fichero con las ubicaciones del archivo" << std::endl;

	std::vector<string> archivos;	
	archivos = videoFiles("C:\Users\Pablo\Desktop\VideosTT");
	
	cout << "Archivos encontrados" << endl;
	for (string fileFolder : archivos) {
		cout << fileFolder << endl;
	}

	cout << "Proceso inciado" << endl;
	for (string fileFolder : archivos) {
		cout << "Archivero a analizar: " << fileFolder << endl;
		startProcess(fileFolder);
		cout << "Proceso de guardado de Frames finalizado" << endl;
	}
	
	cout << "Proceso general finalizado" << endl;
	return 0;
}

void startProcess(const string& path) {

	//Leer el archivero .txt
	std::ifstream myfile(path + "/frames.txt");

	//Vectores para guardar subgrupos e intervaloes
	vector<double> intervalos = {};
	vector<vector<string>> directorios = {};

	//Leer el archivo
	string myline;
	vector<double> timeMilisec;

	while (getline(myfile, myline)) {
		vector<string> aux = getWords(myline, " "); // 3: [inicio, final, nombre_carpeta]

		//Obtener intervalos en enteros
		timeMilisec = getIntervals(aux[0], aux[1]);
		intervalos.push_back(timeMilisec[0]);
		intervalos.push_back(timeMilisec[1]);

		//Obtener los directorios a crear
		directorios.push_back(getGroups(aux[2]));
	}
	myfile.close();

	//Crear los directorios con los grupos obtenidos
	createDirectories(path, directorios);

	//Obtener los frames de los videos
	cout << "Obteniendo los frames" << endl;
	vector<Mat> videoFrames = {};
	vector<int> n_frames = {};

	getFrames(path, videoFrames, n_frames, intervalos);
	cout << n_frames.size()<< intervalos.size() << endl;
	//imshow("Prueba",videoFrames[0]);
	//cv::waitKey(0);

	//Guardar los frames
	cout << "Guardando los frames" << endl;
	saveFrames(videoFrames, n_frames, directorios, path);
	cout << "Proceso finalizado" << endl;
}

//Funcion que regresa lista con todos los directorios encontrados
vector<string> videoFiles(const string file) {
	std::ifstream myfile(file);
	std::string myline;
	std::vector<string> files;

	while (getline(myfile, myline)) {
		// Output the text from the file
		//cout << myText;
		files.push_back(myline);
	}
	myfile.close();
	return files;
}

//Funcion para delimita las palabras
vector<string> getWords(string s, string delimiter) {
	vector<string> res;
	string delim = delimiter;
	string token = "";
	for (int i = 0; i < s.size(); i++) {
		bool flag = true;
		for (int j = 0; j < delim.size(); j++) {
			if (s[i + j] != delim[j]) flag = false;
		}
		if (flag) {
			if (token.size() > 0) {
				res.push_back(token);
				token = "";
				i += delim.size() - 1;
			}
		}
		else {
			token += s[i];
		}
	}
	res.push_back(token);
	return res;
}

//Funcion que regresa cada intervalo
vector<double> getIntervals(string start, string end) {
	
	//Se obtiene arreglos de longitud con [minutos, segundo, milisegundos]
	vector<string> startData = getWords(start, ":");
	vector<string> endData = getWords(end, ":");

	double startMilisecond = (stoi(startData[0]) * 60000) + (stoi(startData[1]) * 1000) + (stoi(startData[2]) * 10);
	double endMilisecond = (stoi(endData[0]) * 60000) + (stoi(endData[1]) * 1000) + (stoi(endData[2]) * 10);

	vector<double> final;
	final.push_back(startMilisecond);
	final.push_back(endMilisecond);

	return final;
}

//Funcion que regresa grupo y subgrupo de cada intervalo
vector<string> getGroups(string group) {

	vector<string> aux;
	vector<string> final;
	aux = getWords(group, "_");
	
	final.push_back(aux[0]);
	final.push_back(group);
	return final;
}


//Funcion para crear intervalos y subgrupos



//Funcion para crear ficheros
void createDirectories(const string path, const vector<vector<string>>& groups) {

	string group;
	string subgroup;
	std::string directoryName1;
	std::string directoryName2;
	for (vector<string> data : groups) {
		group = data[0];
		subgroup = data[1];
		unordered_set<string> set_groups = {}; //Conjunto para guardar carpeta repetidas y no crearlas de nuevo

		if (set_groups.find(group) == set_groups.end()) {
			directoryName1 = path + "/" + group;
			if (createDirectory(directoryName1)) {
				cout << "Folder created successfully." << endl;
				set_groups.insert(group);
			} else {
				cout << "Failed to create folder." << endl;
			}
		}
		directoryName2 = path + "/" + group + "/" + subgroup;
		if (createDirectory(directoryName2)) {
			cout << "Folder created successfully." << endl;
		} else {
			cout << "Failed to create folder." << endl;
		}
	}
}

//Funcion para obtener todos los frames por lapsos de tiempo
void getFrames(const string videoPath, vector<Mat>& frames, vector<int>& n_frames, const vector<double>& interval) {
	
	VideoCapture cap(videoPath + "/video.mp4");
	
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		//return -1
		exit(1);
	}
	//Indice para guiarse
	int	absoluto = 0;
	int i = 0;
	int aux = 0;

	//Obtener diferencias de frame a frame
	double secFrame = 1000 / cap.get(5);
	//Bandera
	bool saveFrame = false;

	//cout << "Diferenica" << secFrame << endl;

	while (1) {
		Mat frame;
		// Capture frame-by-frame
		cap >> frame;
		// If the frame is empty, break immediately
		if (frame.empty()) {
			//if (saveFrame) lapsosFrames.push_back(absoluto);
			break;
		}

		double diferencia = interval[i] - cap.get(0);
		int userInput;
		if (abs(diferencia) <= secFrame / 2) { //Calcula la diferencia
			//cout << abs(diferencia) << ":" << secFrame / 2 << endl;
			//std: cin >> userInput;
			/*if (diferencia < 0) {
				saveFrame = !saveFrame;
			}
			else {
				framesVideo.push_back(frame);
				saveFrame = !saveFrame;
			}*/
			if (saveFrame) {
				//cout << absoluto << endl;
				n_frames.push_back(absoluto);
			}
			saveFrame = !saveFrame;
			i++;
		}

		if (saveFrame) {
			//cout << "frame guaradao" << endl;
			frames.push_back(frame);
			++absoluto;
		}
	}
}

//Funci�n para guardar todos los frames por secuencia
void saveFrames(const vector<Mat> &frames, const vector<int> &n_frames, const vector<vector<string>> &sequence,const string path) {
	int n_group = 0;
	int aux = 0;
	int i = 1;
	string group = sequence[n_group][0];
	string subgroup = sequence[n_group][1];
	string fileName;
	cout << "Funcion para guardar los frames" << endl;
	//while (i < frames.size()-1) {
	for(Mat frame: frames) {
		if (i == (n_frames[n_group]) - 1 ) {

			++n_group; // Si ya alcanzo todos los frames de ese grupo que pase al otro
			aux = 0;

			if (n_group < n_frames.size()) {
				group = sequence[n_group][0];//Renombra los grupos
				subgroup = sequence[n_group][1];
			}

			//cout << "Cambiando grupo" << endl;
			//cout <<  group<< " : " << subgroup << endl;
		}
		//Directorio donde guardar cada frame
		fileName = path + "/" + group + "/" + subgroup + "/" + to_string(aux) + ".jpg";
		//cout << fileName << endl;
		imwrite(fileName, frame);
		aux++;
		i++;
	}
}

//Crear directorio
bool createDirectory(const string& path) {
	if (CreateDirectoryA(path.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS) {
		std::cout << "Directory created or already exists.\n";
		return true;
	}
	else {
		std::cerr << "Failed to create directory!\n";
		return false;
	}
}