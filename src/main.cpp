#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <RDA.hpp>

int main(int argc, char **argv)
{
  std::string line;
  std::vector<TrainData> td,vd;
  const int trainnum = 15000;

  RDA rda(0.000001); //96.7174%

  if(argc < 3){
    std::cerr << "main trainingdata validationdata" << std::endl;
    return -1;
  }

  // read training data
  std::cerr << "begin reading traindata" << std::endl;
  std::ifstream traindata(argv[1]);
  while(std::getline(traindata, line)){
    td.push_back(convert(line));
  }
  std::cerr << "end reading traindata" << std::endl;

  //reading validation data
  std::cerr << "begin reading validationdata" << std::endl;
  std::ifstream validationdata(argv[2]);
  while(std::getline(validationdata, line)){
    vd.push_back(convert(line));
  }
  std::cerr << "end reading validationdata" << std::endl;


  //train
  std::cerr << "begin reading validationdata" << std::endl;
  for(int i=0;i<trainnum;++i){
    rda.update(td[i]);
  }

  std::cerr << "train end" << std::endl;

  //check
  int correct=0;
  for(int i=0;i<vd.size();++i){
    int flag = rda.classify(vd[i].second);
    if(flag == vd[i].first){
      correct++;
    }
  }

  std::cout << (double) correct / (double) vd.size() * 100.0 << std::endl;
  return 0;
}
