import numpy as np
import tensorflow as tf
from typing import Literal
from TimeGAN.utils import extract_time, random_generator_tf, make_tf_dataset #batch_generator

def rnn_cell(module_name: Literal['gru', 'lstm', 'lstmLN'], hidden_dim: int):
    """Return a Keras RNN cell."""
    assert module_name in ['gru', 'lstm', 'lstmLN']
    
    if module_name == 'gru':
        return tf.keras.layers.GRUCell(hidden_dim)
    elif module_name == 'lstm':
        return tf.keras.layers.LSTMCell(hidden_dim)
    elif module_name == 'lstmLN':
        # LayerNorm LSTM is not built-in; standard LSTM as placeholder
        return tf.keras.layers.LSTMCell(hidden_dim)

def build_rnn_model(input_dim, output_dim, module_name, hidden_dim, num_layers, activation='sigmoid', name=None):
    """Build Keras RNN model with stacked cells."""
    cells = [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)]
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(None, input_dim)),
        tf.keras.layers.RNN(cells, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_dim, activation=activation))
    ], name=name)

def timegan(ori_data, parameters):
    """TimeGAN in TF2/Keras."""
    ori_data = np.array(ori_data)
    no, seq_len, dim = ori_data.shape
    ori_time = np.array([seq_len]*no)
    max_seq_len = seq_len

    # Min-Max normalization
    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = (data - min_val) / (max_val - min_val + 1e-7)
        return norm_data, min_val, max_val
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Network parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1.0

    dataset = make_tf_dataset(ori_data, ori_time, batch_size)
    iterator = iter(dataset)  # Get an iterator for your training loop

    # Build models
    embedder = build_rnn_model(dim, hidden_dim, module_name, hidden_dim, num_layers, name="embedder")
    recovery = build_rnn_model(hidden_dim, dim, module_name, hidden_dim, num_layers, name="recovery")
    generator = build_rnn_model(z_dim, hidden_dim, module_name, hidden_dim, num_layers, name="generator")
    supervisor = build_rnn_model(hidden_dim, hidden_dim, module_name, hidden_dim, num_layers, name="supervisor")
    discriminator = build_rnn_model(hidden_dim, 1, module_name, hidden_dim, num_layers, activation=None, name="discriminator")

    # Optimizers
    E_optimizer = tf.keras.optimizers.Adam()
    G_optimizer = tf.keras.optimizers.Adam()
    D_optimizer = tf.keras.optimizers.Adam()

    # Loss functions
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    # -----------------------------
    # Training functions
    # -----------------------------
    @tf.function
    def train_embedder(X_mb):
        with tf.GradientTape() as tape:
            H = embedder(X_mb, training=True)
            X_tilde = recovery(H, training=True)
            E_loss_T0 = tf.sqrt(mse(X_mb, X_tilde))
        grads = tape.gradient(E_loss_T0, embedder.trainable_variables + recovery.trainable_variables)
        E_optimizer.apply_gradients(zip(grads, embedder.trainable_variables + recovery.trainable_variables))
        return E_loss_T0

    @tf.function
    def train_generator_supervised(Z_mb, X_mb):
        with tf.GradientTape() as tape:
            E_hat = generator(Z_mb, training=True)
            H_hat = supervisor(E_hat, training=True)
            H = embedder(X_mb, training=False)
            H_hat_supervise = supervisor(H, training=True)
            G_loss_S = mse(H[:,1:,:], H_hat_supervise[:,:-1,:])
        grads = tape.gradient(G_loss_S, generator.trainable_variables + supervisor.trainable_variables)
        G_optimizer.apply_gradients(zip(grads, generator.trainable_variables + supervisor.trainable_variables))
        return G_loss_S

        
    # @tf.function
    # def train_joint_hybrid(X_mb, Z_mb, itt, threshold=0.15):
    #     # ---------- Pass 1: G + E with a persistent tape (reuse forward; free after) ----------
    #     with tf.GradientTape(persistent=True) as tape:
    #         tf.print(f'training generator and embedding')
    #         # Embedder & recovery
    #         H = embedder(X_mb, training=True)
    #         X_tilde = recovery(H, training=True)

    #         # Generator & supervisor
    #         E_hat = generator(Z_mb, training=True)
    #         H_hat = supervisor(E_hat, training=True)
    #         H_hat_supervise = supervisor(H, training=True)
    #         X_hat = recovery(H_hat, training=True)

    #         # --- Generator losses (use D in inference mode here) ---
    #         # (TF1 used raw logits with sigmoid CE; use from_logits=True)
    #         Y_fake     = discriminator(H_hat, training=False)
    #         Y_fake_e   = discriminator(E_hat, training=False)
    #         G_loss_U   = bce(tf.ones_like(Y_fake),   Y_fake)
    #         G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

    #         # Supervised loss (next-step prediction in latent space)
    #         G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

    #         # Moment matching (stop gradients through X_hat for this part to save memory/compute)
    #         X_hat_ng  = tf.stop_gradient(X_hat)
    #         G_loss_V1 = tf.reduce_mean(
    #             tf.abs(
    #                 tf.sqrt(tf.math.reduce_variance(X_hat_ng, axis=0) + 1e-6) -
    #                 tf.sqrt(tf.math.reduce_variance(X_mb,     axis=0) + 1e-6)
    #             )
    #         )
    #         G_loss_V2 = tf.reduce_mean(
    #             tf.abs(tf.reduce_mean(X_hat_ng, axis=0) - tf.reduce_mean(X_mb, axis=0))
    #         )
    #         G_loss_V  = G_loss_V1 + G_loss_V2

    #         # Final generator loss (mirrors the TF1 weights)
    #         G_loss = G_loss_U + gamma * G_loss_U_e + 100.0 * tf.sqrt(G_loss_S) + 100.0 * G_loss_V

    #         # Embedder loss (reconstruction + small supervision)
    #         E_loss_T0 = mse(X_mb, X_tilde)
    #         E_loss    = 10.0 * tf.sqrt(E_loss_T0) + 0.1 * G_loss_S

    #     # Apply G+E grads
    #     G_vars = generator.trainable_variables + supervisor.trainable_variables
    #     E_vars = embedder.trainable_variables  + recovery.trainable_variables

    #     G_grads = tape.gradient(G_loss, G_vars)
    #     E_grads = tape.gradient(E_loss, E_vars)

    #     G_optimizer.apply_gradients(zip(G_grads, G_vars))
    #     E_optimizer.apply_gradients(zip(E_grads, E_vars))

    #     # IMPORTANT: free the big graph ASAP
    #     del tape

    #     # ---------- Pass 2: Discriminator (less frequent, short-lived tape) ----------
    #     # Match TF1 schedule: update D every other iteration, and only when loss > threshold
    #     if itt % 2 == 0:
    #         with tf.GradientTape() as tape_d:
    #             tf.print(f'training discrim')
    #             # Stop gradients so D doesn’t backprop into G/E during D update
    #             H_sg     = tf.stop_gradient(H)
    #             H_hat_sg = tf.stop_gradient(H_hat)
    #             E_hat_sg = tf.stop_gradient(E_hat)

    #             Y_real   = discriminator(H_sg,     training=True)
    #             Y_fake   = discriminator(H_hat_sg, training=True)
    #             Y_fake_e = discriminator(E_hat_sg, training=True)

    #             D_loss_real   = bce(tf.ones_like(Y_real),   Y_real)
    #             D_loss_fake   = bce(tf.zeros_like(Y_fake),  Y_fake)
    #             D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e)
    #             D_loss        = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    #         if D_loss > threshold:
    #             D_grads = tape_d.gradient(D_loss, discriminator.trainable_variables)
    #             D_optimizer.apply_gradients(zip(D_grads, discriminator.trainable_variables))
    #     else:
    #         D_loss = tf.constant(0.0, dtype=tf.float32)

    #     # Return individual components for logging
    #     return D_loss, G_loss, G_loss_S, G_loss_V, tf.sqrt(E_loss_T0)

    # ---------- Pass 1: Generator + Embedder ----------
    @tf.function
    def train_generator_embedder(X_mb, Z_mb, gamma=1.0):
        tf.print("Training generator and embedder")

        with tf.GradientTape(persistent=True) as tape:
            # Embedder & Recovery
            H = embedder(X_mb, training=True)
            X_tilde = recovery(H, training=True)

            # Generator & Supervisor
            E_hat = generator(Z_mb, training=True)
            H_hat = supervisor(E_hat, training=True)
            H_hat_supervise = supervisor(H, training=True)
            X_hat = recovery(H_hat, training=True)

            # Generator losses
            Y_fake   = discriminator(H_hat, training=False)
            Y_fake_e = discriminator(E_hat, training=False)
            G_loss_U   = bce(tf.ones_like(Y_fake),   Y_fake)
            G_loss_U_e = bce(tf.ones_like(Y_fake_e), Y_fake_e)

            # Supervised loss
            G_loss_S = mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

            # Moment matching
            X_hat_ng = tf.stop_gradient(X_hat)
            G_loss_V1 = tf.reduce_mean(
                tf.abs(tf.sqrt(tf.math.reduce_variance(X_hat_ng, axis=0)+1e-6) -
                    tf.sqrt(tf.math.reduce_variance(X_mb, axis=0)+1e-6))
            )
            G_loss_V2 = tf.reduce_mean(
                tf.abs(tf.reduce_mean(X_hat_ng, axis=0) -
                    tf.reduce_mean(X_mb, axis=0))
            )
            G_loss_V = G_loss_V1 + G_loss_V2

            # Final losses
            G_loss = G_loss_U + gamma * G_loss_U_e + 100.0*tf.sqrt(G_loss_S) + 100.0*G_loss_V
            E_loss_T0 = mse(X_mb, X_tilde)
            E_loss = 10.0*tf.sqrt(E_loss_T0) + 0.1*G_loss_S

        # Apply gradients
        G_vars = generator.trainable_variables + supervisor.trainable_variables
        E_vars = embedder.trainable_variables + recovery.trainable_variables

        G_grads = tape.gradient(G_loss, G_vars)
        E_grads = tape.gradient(E_loss, E_vars)

        G_optimizer.apply_gradients(zip(G_grads, G_vars))
        E_optimizer.apply_gradients(zip(E_grads, E_vars))

        del tape
        return H, H_hat, E_hat, tf.sqrt(E_loss_T0), G_loss, G_loss_S, G_loss_V


    # ---------- Pass 2: Discriminator ----------
    @tf.function
    def train_discriminator(H, H_hat, E_hat, threshold=0.15):
        tf.print("Training discriminator")

        H_sg = tf.stop_gradient(H)
        H_hat_sg = tf.stop_gradient(H_hat)
        E_hat_sg = tf.stop_gradient(E_hat)

        with tf.GradientTape() as tape_d:
            Y_real   = discriminator(H_sg, training=True)
            Y_fake   = discriminator(H_hat_sg, training=True)
            Y_fake_e = discriminator(E_hat_sg, training=True)

            D_loss_real   = bce(tf.ones_like(Y_real),   Y_real)
            D_loss_fake   = bce(tf.zeros_like(Y_fake),  Y_fake)
            D_loss_fake_e = bce(tf.zeros_like(Y_fake_e), Y_fake_e)

            D_loss = D_loss_real + D_loss_fake + gamma*D_loss_fake_e

        if D_loss > threshold:
            D_grads = tape_d.gradient(D_loss, discriminator.trainable_variables)
            D_optimizer.apply_gradients(zip(D_grads, discriminator.trainable_variables))

        return D_loss
    
    def train_joint_step(X_mb, Z_mb, itt, gamma=1.0, threshold=0.15):
        H, H_hat, E_hat, E_loss, G_loss, G_loss_S, G_loss_V = train_generator_embedder(X_mb, Z_mb, gamma)

        # Only update D every 2 iterations
        if itt % 2 == 0:
            D_loss = train_discriminator(H, H_hat, E_hat, threshold)
        else:
            D_loss = tf.constant(0.0, dtype=tf.float32)

        return D_loss, G_loss, G_loss_S, G_loss_V, E_loss


    # -----------------------------
    # Training loops
    # -----------------------------

    import os
    from datetime import datetime

    # Generate timestamp, e.g., "2025-08-18_20-45-30"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a unique checkpoint directory
    checkpoint_dir = f"model_checkpoints/timegan/{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    def save_timegan_models(epoch):
        embedder.save(os.path.join(checkpoint_dir, f"embedder_epoch.keras"))
        recovery.save(os.path.join(checkpoint_dir, f"recovery_epoch.keras"))
        generator.save(os.path.join(checkpoint_dir, f"generator_epoch.keras"))
        supervisor.save(os.path.join(checkpoint_dir, f"supervisor_epoch.keras"))
        discriminator.save(os.path.join(checkpoint_dir, f"discriminator_epoch.keras"))


    # 1. Embedder training
    print("Start Embedding Network Training")
    for itt in range(iterations):
        X_mb, T_mb = next(iterator)
        # X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # print(f'T_mb.shape {T_mb.shape}')
        step_e_loss = train_embedder(X_mb)
        if itt % 100 == 0:
            print(f"Embedding step {itt}/{iterations}, E_loss: {step_e_loss.numpy():.4f}")
            save_timegan_models(itt)
    print("Finish Embedding Training\n")

    # 2. Generator supervised training
    print("Start Generator Supervised Training")
    for itt in range(iterations):
        X_mb, T_mb = next(iterator)
        # X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        # print(f'T_mb shape {T_mb.shape}')
        Z_mb = random_generator_tf(batch_size, z_dim, T_mb, max_seq_len)
        step_g_loss_s = train_generator_supervised(Z_mb, X_mb)
        if itt % 100 == 0:
            print(f"Supervised step {itt}/{iterations}, G_loss_S: {step_g_loss_s.numpy():.4f}")
            save_timegan_models(itt)
    print("Finish Generator Supervised Training\n")

    # 3. Joint training
    print("Start Joint Training")
    for itt in range(2*iterations):
        X_mb, T_mb = next(iterator)
        # Generator + Embedder update twice, discriminator update one
        # X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator_tf(batch_size, z_dim, T_mb, max_seq_len)
        step_d_loss, step_g_loss_u, step_g_loss_s, step_g_loss_v, step_e_loss_t0 = train_joint_step(X_mb, Z_mb, itt)

        if itt % 1 == 0:
            print('step: '+ str(itt) + '/' + str(2*iterations) + 
                    ', d_loss: ' + str(np.round(step_d_loss,4)) + 
                    ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) + 
                    ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) + 
                    ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) + 
                    ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4)))
        if itt % 10 == 0:
            save_timegan_models(itt)

    # -----------------------------
    # Generate synthetic data
    # -----------------------------
    Z_mb = random_generator_tf(no, z_dim, ori_time, max_seq_len)
    H_hat = supervisor(generator(Z_mb, training=False), training=False)
    X_hat = recovery(H_hat, training=False).numpy()
    generated_data = [(X_hat[i,:ori_time[i],:] * max_val + min_val) for i in range(no)]

    return generated_data

