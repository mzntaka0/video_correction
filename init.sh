conda install -c https://conda.anaconda.org/menpo opencv3
brew install cmake
brew install boost-python --with-python3
mv /usr/local/opt/boost-python/lib/libboost_python3-mt.dylib /usr/local/opt/boost-python/lib/libboost_python-mt.dylib
git clone https://github.com/shunsukeaihara/colorcorrect.git
cd colorcorrect
python setup.py build
python setup.py install
cd ..
rm -rf colorcorrect

pip install -r requirements.txt
