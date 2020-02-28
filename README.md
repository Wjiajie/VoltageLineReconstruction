## 一个电线三维重建的例子


![result.PNG](https://i.loli.net/2020/02/28/aoZsw5TndvHL6fi.png)


### 环境：
* vs2017
* opencv 3
* Eigen 3.4
* ceres-solver 1.14

### 流程
1. 地面特征提取匹配，求解相机位姿（代码不包括该内容）。
2. 电线提取。
3. 利用对极约束和电线位置关系提取电线中的匹配点。
4. DLT法求电线3D点初值。
5. ceres 优化3D点和相机位姿。

### 测试数据
[百度云链接](https://pan.baidu.com/s/1faPRjVjBSTf9XgBR4YFF-g)

提取码：v83d
