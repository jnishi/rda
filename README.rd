#AdaGrad+ RDA

*ビルド方法
$ ./waf configure
$ ./waf

* 実行方法
$ build/src/main trainingdata(学習データ) validationdata(検証データ)
96.7174

* 学習・検証データフォーマット
main はlibSVM形式のデータを読み込みます。
libSVMで用いられる形式のデータを読み込む関数フォーマットは以下のとおりです。
  (BNF-like representation)
  
  <class> .=. +1 | -1 
  <feature> .=. integer (>=1)
  <value> .=. real
  <line> .=. <class> <feature>:<value><feature>:<value> ... <feature>:<value>
