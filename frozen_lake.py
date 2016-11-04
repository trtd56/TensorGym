# coding:utf-8


# --- import modules ---
import gym
import numpy as np
import random

# --- import deep learning modules ---
import tensorflow as tf

### environment ###
ENV_NAME    = "FrozenLake-v0"
N_EPISODES  = 100
N_INPUT     = 16
N_ACT       = 4

### agent ###
GAMMA       = 0.99
EPS         = 0.1



def main():
    env = gym.make(ENV_NAME)

    tf.reset_default_graph()    # 現在のグラフをリセット

    # 行動選択の為のネットワーク
    inputs1 = tf.placeholder(shape=[1,N_INPUT],dtype=tf.float32)     # 入力は4×4の16
    W = tf.Variable(tf.random_uniform([N_INPUT, N_ACT], 0, 0.01))    # NNの重み
    Qout = tf.matmul(inputs1,W)                                 # Q値の出力
    predict = tf.argmax(Qout,1)                                 # Q値の最大を取る

    # 学習部分の定義
    nextQ = tf.placeholder(shape=[1, N_ACT],dtype=tf.float32)           # 次の行動(上下左右)のQ値
    loss = tf.reduce_sum(tf.square(nextQ - Qout))                       # 出力Q値の平均二乗誤差を取る
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)      # 学習の最適化手法の定義
    updateModel = trainer.minimize(loss)                                # モデルの学習

    #　ネットワークの学習
    init = tf.initialize_all_variables()    # 初期化

    jList = []      # ステップ数のリスト
    rList = []      # 報酬のリスト
    with tf.Session() as sess:
        sess.run(init)
        for i in range(N_EPISODES):
            # 環境の初期化
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 99:
                j += 1
                a, allQ = sess.run([predict, Qout],feed_dict={inputs1:np.identity(N_INPUT)[s:s+1]})
                if np.random.rand(1) < EPS:
                    a[0] = env.action_space.sample()
                s1, r, d, _ = env.step(a[0])
                Q1 = sess.run(Qout, feed_dict={inputs1:np.identity(N_INPUT)[s1:s1+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = r + GAMMA * maxQ1
                _, W1 = sess.run([updateModel, W],feed_dict={inputs1:np.identity(N_INPUT)[s:s+1],nextQ:targetQ})
                rAll += r
                s = s1
                if d == True:
                    e = 1./((i/50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
    print "Percent of succesful episodes: " + str(sum(rList)/N_EPISODES) + "%"


if __name__ == '__main__':
    main()
