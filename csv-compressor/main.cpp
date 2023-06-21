#include <iostream>
#include <filesystem>
#include <fstream>

using namespace std;
using namespace std::filesystem;

#define var auto
#define in :

int main(int argv, char** args) {
    path s("/Users/dayo/linux/주제탐구/archive (1)/musicnet/musicnet/train_labels/");
    ofstream os;
    os.open(s / path("../train_labels.csv"));
    int ii = 0;

    for(var x in directory_iterator(s)) {
        string cache = x.path();
        cout << "Running: " << x.path() << endl;
        if(!x.is_directory() && cache.substr(cache.size() - 4) == ".csv") {
            ifstream fs;
            fs.open(cache);

            if(ii) {
                string ss;
                getline(fs, ss);//read one line which to ignore.
            }
            ii = 1;
            int mode = 0;
            char* c = new char[1];
            while(!fs.eof()) {
                fs.read(c, 1);
                if(*c == '\n') mode = 0;
                else if(*c == ',') mode++;
                if(mode < 6)
                    os << *c;
            }
            fs.close();
        }
    }
    os.close();
    return 0;
}
