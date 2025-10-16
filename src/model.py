"""
ML 모델 정의
LSTM, GRU, Transformer 기반 수화 인식 모델
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


class LSTMSignLanguageModel(Model):
    """LSTM 기반 수화 인식 모델"""
    
    def __init__(self, config):
        super(LSTMSignLanguageModel, self).__init__()
        
        self.config = config
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config['dropout']
        num_classes = config['num_classes']
        
        # LSTM 레이어
        self.lstm_layers = []
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)  # 마지막 레이어만 False
            self.lstm_layers.append(
                layers.LSTM(hidden_dim, 
                          return_sequences=return_sequences,
                          dropout=dropout,
                          recurrent_dropout=dropout)
            )
        
        # Fully Connected 레이어
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        """
        순전파
        
        Args:
            inputs: (batch_size, sequence_length, input_dim)
            training: 학습 모드 여부
        
        Returns:
            outputs: (batch_size, num_classes)
        """
        x = inputs
        
        # LSTM 레이어 통과
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # FC 레이어
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        
        return outputs


class GRUSignLanguageModel(Model):
    """GRU 기반 수화 인식 모델"""
    
    def __init__(self, config):
        super(GRUSignLanguageModel, self).__init__()
        
        self.config = config
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        num_layers = config['num_layers']
        dropout = config['dropout']
        num_classes = config['num_classes']
        
        # GRU 레이어
        self.gru_layers = []
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            self.gru_layers.append(
                layers.GRU(hidden_dim,
                         return_sequences=return_sequences,
                         dropout=dropout,
                         recurrent_dropout=dropout)
            )
        
        # FC 레이어
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = inputs
        
        for gru_layer in self.gru_layers:
            x = gru_layer(x, training=training)
        
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        
        return outputs


class TransformerSignLanguageModel(Model):
    """Transformer 기반 수화 인식 모델"""
    
    def __init__(self, config):
        super(TransformerSignLanguageModel, self).__init__()
        
        self.config = config
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        num_heads = config.get('num_heads', 4)
        num_layers = config['num_layers']
        dropout = config['dropout']
        num_classes = config['num_classes']
        
        # Input projection
        self.input_projection = layers.Dense(hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            )
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # FC layers
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        # Input projection
        x = self.input_projection(inputs)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        
        # Pooling
        x = self.global_pool(x)
        
        # FC layers
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        outputs = self.output_layer(x)
        
        return outputs


class PositionalEncoding(layers.Layer):
    """Positional Encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Positional encoding 생성
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]


class TransformerEncoderLayer(layers.Layer):
    """Transformer Encoder Layer"""
    
    def __init__(self, d_model, num_heads, dropout):
        super(TransformerEncoderLayer, self).__init__()
        
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, 
                                                   key_dim=d_model)
        self.ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, x, training=False):
        # Multi-head attention
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


def create_model(config):
    """
    설정에 따라 모델 생성
    
    Args:
        config: 모델 설정 딕셔너리
    
    Returns:
        model: Keras 모델
    """
    model_type = config['type'].lower()
    
    if model_type == 'lstm':
        model = LSTMSignLanguageModel(config)
    elif model_type == 'gru':
        model = GRUSignLanguageModel(config)
    elif model_type == 'transformer':
        model = TransformerSignLanguageModel(config)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    return model


def compile_model(model, config):
    """
    모델 컴파일
    
    Args:
        model: Keras 모델
        config: 학습 설정 딕셔너리
    
    Returns:
        compiled_model: 컴파일된 모델
    """
    optimizer_name = config['optimizer'].lower()
    
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'])
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=config['learning_rate'])
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=config['learning_rate'])
    else:
        raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
    
    model.compile(
        optimizer=optimizer,
        loss=config['loss'],
        metrics=['accuracy']
    )
    
    return model


# 테스트 코드
if __name__ == "__main__":
    # 테스트 설정
    config = {
        'type': 'LSTM',
        'input_dim': 63,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'num_classes': 10
    }
    
    training_config = {
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss': 'categorical_crossentropy'
    }
    
    # 모델 생성 및 컴파일
    model = create_model(config)
    model = compile_model(model, training_config)
    
    # 샘플 데이터로 테스트
    sample_input = np.random.rand(32, 30, 63)  # (batch, seq_len, features)
    output = model(sample_input, training=False)
    
    print("모델 출력 shape:", output.shape)
    print("모델 요약:")
    model.build(input_shape=(None, 30, 63))
    model.summary()
