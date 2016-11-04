# coding:utf-8


import gym
import numpy as np
import random
import tensorflow as tf


ENV_NAME        = "CartPole-v0" # 環境
N_EPISODES      = 10000         # 最大エピソード数
N_INPUT         = 4             # 入力の次元数
GAMMA           = 0.99          # 割引率
EPS             = 0.1           # 初期epsilon
N_HIDDEN        = 10            # 隠れ層のユニット数
LR              = 1e-2          # 学習率
N_BATCH         = 50            # バッチサイズ
RENDER_INTERVAL = 500           # 映像の表示感覚

def discount_rewards(r):
    """
    エピソードが長く続くほど大きな報酬にする
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    env = gym.make(ENV_NAME)

    tf.reset_default_graph()

    # ネットワークの定義
    observations = tf.placeholder(tf.float32, [None, N_INPUT] , name="input_x")
    W1 = tf.get_variable("W1", shape=[N_INPUT, N_HIDDEN],
                            initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(tf.matmul(observations,W1))
    W2 = tf.get_variable("W2", shape=[N_HIDDEN, 1],
                            initializer=tf.contrib.layers.xavier_initializer())
    score = tf.matmul(layer1,W2)
    probability = tf.nn.sigmoid(score)

    # 方策計算のネットワーク
    tvars = tf.trainable_variables()
    input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
    advantages = tf.placeholder(tf.float32,name="reward_signal")

    # loss計算部分
    loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
    loss = -tf.reduce_mean(loglik * advantages)
    newGrads = tf.gradients(loss, tvars)

    # 最適化手法
    adam = tf.train.AdamOptimizer(learning_rate=LR)
    W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
    W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
    batchGrad = [W1Grad, W2Grad]
    updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

    xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 1
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        observation = env.reset() # 環境の初期化

        # 勾配の初期化
        gradBuffer = sess.run(tvars)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        while episode_number <= N_EPISODES:

            # いい結果が出た時のみ映像を表示する
            if episode_number % RENDER_INTERVAL == 0:
                env.render()

            x = np.reshape(observation,[1,N_INPUT])                     # 環境をNNに入力できるように変換
            tfprob = sess.run(probability,feed_dict={observations: x})  # 方策を求める
            action = 1 if np.random.uniform() < tfprob else 0           # 初期アクション(ランダム)
            xs.append(x)                                                # 環境リストに追加
            y = 1 if action == 0 else 0                                 # ダミーラベル
            ys.append(y)                                                # 行動履歴の保存
            observation, reward, done, info = env.step(action)          # 環境を変化させる
            reward_sum += reward                                        # 報酬を合計していく
            drs.append(reward)                                          # 過去の報酬を覚えておく
            if done:
                episode_number += 1

                # 今回のエピソードの記憶を別に移して、記憶をリセットする
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                tfp = tfps
                xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]

                discounted_epr = discount_rewards(epr)  # 長く続くほど報酬を高くする
                discounted_epr -= np.mean(discounted_epr)   # 標準化：平均を引く
                discounted_epr /= np.std(discounted_epr)    # 標準化：標準偏差で割る

                # このエピソードでの勾配を求める
                tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad

                # バッチサイズ毎にネットワークの更新を行う
                if episode_number % N_BATCH == 0:
                    sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                    # エピソード報酬の
                    reward_avg = reward_sum/N_BATCH
                    print 'episode %i : average reward %f' % (episode_number, reward_avg)

                    # 報酬平均が200を超えると学習完了
                    if reward_avg > 200:
                        break

                    reward_sum = 0
                observation = env.reset()   # 環境の初期化

        print "Complete!!"

if __name__ == '__main__':
    main()
