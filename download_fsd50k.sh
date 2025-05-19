wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z01
wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z02
wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z03
wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z04
wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z05
wget https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip

zip -s 0 FSD50K.dev_audio.zip --out unsplit.zip

unzip unsplit.zip

wget https://zenodo.org/records/4060432/files/FSD50K.metadata.zip
unzip FSD50K.metadata.zip
mv FSD50K.metadata FSD50K.dev_audio
