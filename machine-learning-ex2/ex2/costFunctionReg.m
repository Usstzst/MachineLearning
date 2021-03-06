function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%s theta=length(theta);

% z=X*theta;
% h=sigmoid(z);
% logisf=(-y).*log(h)-(1-y).*log(1-h);
% 
% J=((1/m)*sum(logisf))+(lambda/(2*m))*sum(theta.^2);
% 
% n=length(theta);
% 
%  grad(1)=1/m.*(sum((X(1,:))'*h-(X(1,:))'*y));
% % grad(1)=1/m.*(X'*h-X'*y);
% grad=(1/m).*((X'*h-X'*y)+theta*lambda);  % This gives correct answer but ...
% 
% for i=2:n
% 	grad(i) = (1/m) * (h-y)' * X(:,i) + (lambda / m) * theta(i);
% end

z=X*theta;
h=sigmoid(z);
logisf=(-y).*log(h)-(1-y).*log(1-h);

tempTheta = theta;
tempTheta(1) = 0;

J = (1 / m) * sum(logisf) + (lambda / (2 * m))*sum(tempTheta.^2);

error = h - y;
grad = (1 / m) * (X' * error) + (lambda/m)*tempTheta;



% =============================================================

end
