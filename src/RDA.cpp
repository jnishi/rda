#include <cmath>
#include "RDA.hpp"

RDA::RDA(double l):lambda(l){
  
}

double RDA::calcInnerProduct(SparseVector v)
{
  double result = 0.0;

  //inner product
  for(SparseVector::const_iterator dim=v.begin(); dim != v.end();++dim){
    std::map<int, double>::const_iterator ret = SeparationPlain.find(dim->first);
    if(ret != SeparationPlain.end()){
      result += ret->second * dim->second;
    }
  }
  return result;
}

TrainLabel RDA::classify(SparseVector v)
{
  double value = calcInnerProduct(v);
  if(value < 0.0){
    return NEGATIVE;
  }else{
    return POSITIVE;
  }
}

void RDA::update(TrainData t)
{
  //ヒンジロスが0の場合は劣微分は0になるので学習にならない
  double loss = 1.0 - t.first * calcInnerProduct(t.second);
  if(loss <= 0.0){
    return;
  }

  //update gradient
  numTrain++;
  for(SparseVector::const_iterator i=t.second.begin();i!=t.second.end();++i){
    int feature = i->first;
    double value = i->second;

    SumOfGradients[feature] -= t.first * value;
    SumOfSquaredGradients[feature] += value * value;

    //sign(u_{t,i})
    int sign = SumOfGradients[feature] > 0 ? 1 : -1;
    //|u_{t,i}|/t - \lambda
    double meansOfGradients = sign * SumOfGradients[feature] / (double)numTrain - lambda;
    if(meansOfGradients < 0.0){
      // x_{t,i} = 0 
      SeparationPlain.erase(feature);
    }else{
      // x_{t,i} = sign(- u_{t,i}) * \frac{\eta t}{\sqrt{G_{t,ii}}}(|u_{t,i}|/t - \lambda)
      SeparationPlain[feature] = -1 * sign * (double)numTrain * meansOfGradients / sqrtf(SumOfSquaredGradients[feature]);
    }    
  }

  return;
}
