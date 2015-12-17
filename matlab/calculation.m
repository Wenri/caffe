% input1: input vector
% input2: size of layer matrix
% input3: activation function, 1: 'sigmoid'  2: 'tanh'
% input4: w
% out1: circulation matrix
% out2: forward propagation result
% out3: backward propagation result

function [circul_matrix, FP, BP] = calculation(data, m, n, activation, w)
fprintf('1./n');
% Step 1: build the matrix
matrix_row = min(m,n) ;
matrix_col = max(m,n) ;
r = rand(1,matrix_row)' ;
circul_matrix = gallery('circul',r)' ;
circul_matrix = [circul_matrix, circul_matrix(: , 1 : (matrix_col - matrix_row))] ;


% Step 2: Forward propagation
dft_r = dft(r') ;
dft_x = dft(data') ;
Rx = idft((dft_r .* dft_x)') ;
if activation == 1
    hx = sigmoid(Rx) ;
elseif activation == 2
    hx = tanh(Rx) ;
end
FP = hx ;

% Step 3: Backward propagation
% Part 1: s_(rev(x))
rev_x = flipdim(data,1) ;
rev_x_end = rev_x(end) ;
s_rev_x = [rev_x_end ; rev_x(1:end-1)] ;
% Part 2: w * T'(r o x))
if activation == 1
    dhx = hx .* (1 - hx) ;
elseif activation == 2
    dhx = 1 - hx.^2 ;
end

dft_s_rev_x = dft(s_rev_x') ;
dft_wT_rox = dft((w .* dhx)') ;
BP = idft((dft_s_rev_x .* dft_wT_rox)') ;

end

% DFT function
function xk = dft(xn) % 
N = length(xn) ;
n = [0:N-1] ;
k = [0:N-1] ;
Wn = exp(-j*2*pi/N) ; 
nk = n'*k ;
Wnnk = Wn.^nk ;
xk = xn*Wnnk ;
xk = xk' ;
end

function xn = idft(xk)
N = length(xk) ;
n = [0:N-1] ;
k = [0:N-1] ;
Wn = exp(-j*2*pi/N) ; 
nk = n'*k ;
Wnnk = Wn.^(-nk) ;
xn = xk*Wnnk/N ;
xn = real(xn') ;
end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function tanh = tanh(x)
    tanh = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x)) ;
end