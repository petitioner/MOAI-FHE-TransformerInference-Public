# MOAI

# 1. Install
Install the third party globally.
```
cd thirdparty/SEAL-4.1-bs
// if build exist, run
// rm -rf build
cmake -S . -B build
cmake --build build
sudo cmake --install build
```
# 2. Go to main folder and Run
```
cmake -S . -B build
cd build
make
./test
```

---
# uninstall thirdparty
```
cd thirdparty/SEAL-4.1-bs/build
sudo checkinstall
dpkg -r build
```
