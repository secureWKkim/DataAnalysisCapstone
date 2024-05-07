# 주관적 감상에 기반한 나만의 음악 장르 분류 (Capstone design 2020-1)

## Overview
* Needs, problems<br>
발상의 시작은 나 자신의 음악 취향에 대한 오랜 관찰 결과였다. 좋아하는 곡들 사이에 일관된 특징 몇 가지가 있었고, 이 특징을 중심으로 내가 좋아하는 음악의 분류가 가능했다. 그러나 이것은 주관적, 직관적으로만 느끼는 부분에 불과했다. 명확한 실체가 없었다. 객관적, 구체적으로 어떤 수치적, 과학적 특징이 나로 하여금 그 음악들이 서로 한 유형을 공유한다고 느끼게 하는 건지 알아보고 싶었다.
뿐만 아니라 최근 발매된 곡들은 한 가지 장르만으로 맞아 떨어지지 않고, 여러 가지 장르가 복합적으로 어우러진 특징을 보이기 때문에 기존의 장르 체계가 절대적인 것이라 볼 수 없다고 생각했다. 때문에 나의 주관적 감상에 따라 음악을 분류한 것 또한 음악 장르 분류로 간주할 수 있다고 생각했다. 따라서 데이터 분석과 신호 처리 지식을 이용하여 이 음악들을 구분하게 하는 특징을 마이닝해보기로 했다.

* Goals, objectives (evaluation)
    - 정성적: 유사도 지표와 클러스터링 기법을 바탕으로 군집화가 가장 잘 되는 특징 벡터 및 군집 분석 방법을 발굴한다.
    - 정량적: 군집 분석 모델 평가 지표에서 정확도가 60% 이상 되는 것을 목표로 한다.

* 활용한 도구 및 방법
    - 구간 파악 및 설정: Wavepad Audio Editor(소프트웨어), librosa 라이브러리
    - 음악 데이터 전처리: pydub 라이브러리, homebrew 패키지 지원 ffmpeg
    - 음악 데이터 특징 추출: librosa 라이브러리
    - 데이터 분석: K-means Clustering, Agglomerative Clustering, Spectral Clustering 기법 (scikit learn)
    - 데이터 시각화: seaborn, matplotlib
    - 군집 모델 평가: Advanced Rand Index, Fowlkes-Mallows Score, Normalized Mutual Information(NMI), Adjusted Mutual Information(AMI)
    - 사용한 음악 데이터 목록: 맨 아래 결과 보고서 링크 참고

## Results
### Main Code
```
def vector_split(v, n_split):
    v=np.array(v)
    length=len(v)//n_split
    res=[]
    for i in range(n_split-1):
        res.append(v[i*length:(i+1)*length])
    res.append(v[(i+1)*length:])
    return np.array(res)

def split_mean(v, n_split):
    splitted_vector=vector_split(v, n_split)
    res=[]
    for i in splitted_vector:
        res.append(i.mean())
    return np.array(res)

def mfccs(songname):
    """
    mfccs = librosa.feature.mfcc(songname, sr=44100, n_mfcc=10)
    mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    """
    mfccs = librosa.feature.mfcc(songname, sr=44100, n_mfcc=20)
    mfccs = split_mean(mfccs, 10)
    return list(mfccs)

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

def centroid(songname):
    song_sc = normalize(librosa.feature.spectral_centroid(songname,sr=sr)[0])
    #frames = range(len(song_sc))
    #t = librosa.frames_to_time(frames)
    song_sc_data = split_mean(song_sc,10)
    return list(song_sc_data)
    """
    # normalize for visualization purposes
    librosa.display.waveplot(songname,sr=sr,alpha=0.4)
    plt.plot(t,song_sc,color='r')
    """

def mel(songname):
    name_stft = np.abs(librosa.stft(songname))
    name_mel = librosa.feature.melspectrogram(S=name_stft**2)
    """
    librosa.display.specshow(librosa.amplitude_to_db(name_mel), x_axis='time', y_axis='log')
    plt.colorbar()
    """
    new = librosa.amplitude_to_db(name_mel)
    new = split_mean(new, 10)
    return list(new)

def contrast(songname):
    name_cont = librosa.feature.spectral_contrast(songname, sr=sr)
    #plt.imshow(normalize(name_cont, axis=1), aspect='auto', origin='lower', cmap='coolwarm')
    name_cont_data = split_mean(name_cont,7)
    return list(name_cont_data)

def tempogram(songname):
    onset_env = librosa.onset.onset_strength(songname, sr=sr, hop_length=200, n_fft=2048)
    name_S = librosa.stft(onset_env, hop_length=1, n_fft=512)
    name_fourier_tempogram = np.absolute(name_S)
    # librosa.display.specshow(name_fourier_tempogram, sr=sr, hop_length=hop_length, x_axis='time')
    name_ft_data = split_mean(name_fourier_tempogram,10)
    return list(name_ft_data)


def writefile(songlist):
    with open("heroic.csv", 'w', newline='') as f:
        w = csv.writer(f)
        for song in songlist:
            temp = ['heroic']
            temp.extend(mfccs(song))
            temp.extend(mel(song))
            temp.extend(centroid(song))
            temp.extend(contrast(song))
            temp.extend(tempogram(song))
            w.writerow(temp)
```
- vector_split, split_mean: 반환되길 원하는 벡터 및 차원 수를 인자로 받아 그에 맞는 차원의 벡터를 반환해준다. vector_split에서 차원 수에 맞게 기존의 벡터를 쪼개 단위별로 묶어 벡터를 반환한다. 그러면 split_mean에서 쪼개진 각 원소별로 평균값을 구해 벡터의 새로운 원소로 삼는다.
- mfccs: librosa 라이브러리 속 feature의 mfcc 메서드를 이용해 20차원의 mfcc 계수를 반환한다. 이를 split_mean 함수를 이용해 10차원 벡터로 반환한다.
- normalize: 사이킷런의 전처리 기능을 이용해 입력 인자로 받은 벡터를 정규화해주는 함수다.
- centroid: librosa 라이브러리 속 feature의 spectral_centroid 메서드를 이용해 입력 인자로 받은 음악 데이터의 spectral centroid 값을 10차원 벡터로 반환한다.
- mel: 입력으로 받은 음악 데이터를 librosa의 stft 함수를 이용해 short time fourier transform 처리한다. 이에 절댓값을 취한 뒤, 각 값을 제곱한 값을 librosa.feature의 melspectrogram 메서드의 입력으로 넘긴다. 이를 amplitude_to_db 함수를 이용해 데시벨 단위로 바꾼 뒤 10차원 벡터로 바꿔 반환한다.
- contrast: librosa.feature의 spectral_contrast 메서드를 이용해 spectral contrast 값을 7차원 벡터로 반환한다. spectral_contrast 메서드에서 기본으로 반환하는 벡터의 차원이 7차원 짜리이므로 10차원 이상의 벡터로 만들 수 없다.
- tempogram: 우선 librosa.onset의 onset_strength 메서드를 이용해 인자로 받은 음악 데이터의 onset envelope의 numpy 배열을 만든다. 이를 short time fourier transform 처리한 뒤 이에 절댓값을 취한 것이 tempo 데이터의 벡터다.
- writefile: 위의 함수를 이용해 추출할 수 있는 모든 피처를 새로운 csv 파일에 작성한다. 다수의 음악 데이터 처리를 위해 만든 함수다.<br>
### Result Visualization Examples (2차원 벡터에 대한 시각화 결과)
- K-means Clustering<br>
데이터를 k개의 클러스터로 나눈 뒤 할당된 클러스터의 평균과 포함된 데이터들의 거리 제곱합이 최소가 되게 한다.
![image](https://user-images.githubusercontent.com/48075848/86555583-bb2efe80-bf8b-11ea-8fc0-c37a23631858.png)
- Agglomerative Clustering<br>
계층적인 방법으로 비슷한 클러스터를 합친다.
![image](https://user-images.githubusercontent.com/48075848/86555604-ca15b100-bf8b-11ea-881a-b5d410ae91ef.png)
- Spectral Clustering<br>
클러스터를 구성하는 노드의 연결성에 기반하여 연결 그래프를 생성하고 데이터 포인트를 그룹화한다.
![image](https://user-images.githubusercontent.com/48075848/86555665-f5989b80-bf8b-11ea-879f-5f13735ef1f1.png)
### Model Evaluation
![image](https://user-images.githubusercontent.com/48075848/86529264-4fe12000-beea-11ea-8771-a03fe40d5b45.png)
![image](https://user-images.githubusercontent.com/48075848/86529266-5c657880-beea-11ea-9486-e0d4a1118882.png)
![image](https://user-images.githubusercontent.com/48075848/86529275-6d15ee80-beea-11ea-9f35-b1d28cf9334b.png)


## Conclusion
벡터 종류 관점에서도, 클러스터링 기법 관점에서 눈에 띄게 좋은 수치 없이 서로 비슷한 수치로 결과가 나왔다. 1에 가까울수록 좋은 지표들은 다 압도적으로 0에 가깝게 나왔고, 어떠한 분석 방법의 조합이든 다 분산이 너무 낮거나, 한 레이블로의 쏠림 현상이 심하게 나타났다. heroic의 경우 한 가지 경우를 제외하곤 대체로 분산이 높게 나왔다. 모든 경우에 대해 과반 이상으로 매칭이 된 레이블이 있었다.
따라서 제언에 다소 어려움이 있다. 다만 음악에서 느껴지는 심상, 분위기를 데이터로 추출하고 분석, 분류하는 일이 결코 만만하지 않다는 것을 깨달았다. 더 일반화하여 주관적이고 추상적인 생각을 검증, 분석 가능한 데이터로 만드는 게 매우 어려운 일임을 깨달았다. 따라서 좀더 객관적이고 검증 가능한 주제 설정의 중요성과 필요성을 절감했다.
<br><br>
분석 결과가 잘 나올 경우 이 프로젝트를 표준적인 장르 분류 체계가 아닌 음악의 심상, 분위기에 따라 음악을 분류하는 일의 단초로 삼고 이에 대한 데이터를 늘려 좀더 일반화된 어플리케이션으로 발전시킬 수 있다는 제언을 하고 싶다.

## Reports
* [아이디어 전개 및 탐구 기록](https://thewayaboutme.tistory.com/category/%EC%88%98%EC%97%85%20%ED%95%84%EA%B8%B0/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%EC%BA%A1%EC%8A%A4%ED%86%A4%EB%94%94%EC%9E%90%EC%9D%B8)
* [최종 발표 자료](https://docs.google.com/presentation/d/10mO6CmG-Tdpx8Uq73ryJRTtjRkP_OmO1/edit?usp=sharing&ouid=118145286936913978381&rtpof=true&sd=true)
