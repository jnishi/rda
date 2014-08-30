/*
  コメント中の表記は以下の論文を元にしている
  Duchi, Hazan, and Singer.
  Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
  
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>

enum TrainLabel{
  POSITIVE=1, NEGATIVE=-1
};

typedef std::vector<std::pair<int, double> > SparseVector;
typedef std::pair<TrainLabel, SparseVector> TrainData;

/*
  libSVMで用いられる形式のデータを読み込む関数フォーマットは以下のとおり
  (BNF-like representation)
  
  <class> .=. +1 | -1 
  <feature> .=. integer (>=1)
  <value> .=. real
  <line> .=. <class> <feature>:<value><feature>:<value> ... <feature>:<value>

*/
TrainData convert(std::string line)
{
  SparseVector result;
  std::istringstream parsed_line(line);
  std::string flag;
  parsed_line >> flag;
  TrainLabel label = flag == "+1" ? POSITIVE : NEGATIVE;  
  
  std::string token;
  while(parsed_line >> token){
    std::string::size_type pos =  token.find(":");
    token.replace(pos, 1, " ");
    std::istringstream iss(token);
    int dim;
    double value;
    iss >> dim >> value;
    result.push_back(std::make_pair(dim, value));
  }
  std::sort(result.begin(), result.end());
  return make_pair(label, result);
}

class RDA {
public:
  RDA(double l);
  TrainLabel classify(SparseVector v);
  void update(TrainData t);

private:
  double calcInnerProduct(SparseVector sv);
  double lambda; //正則化項の強さのパラメータ
  int numTrain; // 学習の回数、論文中はt
  std::map<int, double> SumOfGradients;  // \sum^t_{\tau}g_{\tau}:劣微分ベクトル列の和
  std::map<int, double> SumOfSquaredGradients; //Gが対角行列の時の外積は劣微分ベクトルの二条和で表される
  std::map<int, double> SeparationPlain; //学習した分離平面、論文中はx
};
