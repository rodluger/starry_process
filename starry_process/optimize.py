import theano
import theano.tensor as tt


def Adam(cost, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    """
    https://gist.github.com/Newmu/acb738767acb4788bac3
    """
    updates = []
    grads = tt.grad(cost, params)
    i = theano.shared(np.array(0.0, dtype=theano.config.floatX))
    i_t = i + 1.0
    fix1 = 1.0 - (1.0 - b1) ** i_t
    fix2 = 1.0 - (1.0 - b2) ** i_t
    lr_t = lr * (tt.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.0)
        v = theano.shared(p.get_value() * 0.0)
        m_t = (b1 * g) + ((1.0 - b1) * m)
        v_t = (b2 * tt.sqr(g)) + ((1.0 - b2) * v)
        g_t = m_t / (tt.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


def NAdam(cost, params, lr=0.002, b1=0.9, b2=0.999, e=1e-8, sd=0.004):
    """https://github.com/keras-team/keras/blob/master/keras/optimizers.py
    """
    updates = []
    grads = tt.grad(cost, params)
    i = theano.shared(np.array(0.0, dtype=theano.config.floatX))
    i_t = i + 1.0

    # Warm up
    m_schedule = theano.shared(np.array(1.0, dtype=theano.config.floatX))
    momentum_cache_t = b1 * (1.0 - 0.5 * (tt.pow(0.96, i_t * sd)))
    momentum_cache_t_1 = b1 * (1.0 - 0.5 * (tt.pow(0.96, (i_t + 1) * sd)))
    m_schedule_new = m_schedule * momentum_cache_t
    m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
    updates.append((m_schedule, m_schedule_new))

    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.0)
        v = theano.shared(p.get_value() * 0.0)

        g_prime = g / (1.0 - m_schedule_new)
        m_t = b1 * m + (1.0 - b1) * g
        m_t_prime = m_t / (1.0 - m_schedule_next)
        v_t = b2 * v + (1.0 - b2) * tt.sqr(g)
        v_t_prime = v_t / (1.0 - tt.pow(b2, i_t))
        m_t_bar = (1.0 - momentum_cache_t) * g_prime + (
            momentum_cache_t_1 * m_t_prime
        )

        updates.append((m, m_t))
        updates.append((v, v_t))

        p_t = p - lr * m_t_bar / (tt.sqrt(v_t_prime) + e)
        new_p = p_t
        updates.append((p, new_p))

    updates.append((i, i_t))

    return updates
