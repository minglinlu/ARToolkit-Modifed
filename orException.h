#ifndef __OR_EXCEPTION__
#define __OR_EXCEPTION__

#include<string>
#include <opencv2/core/core.hpp>

namespace cvar{

// General Exception
class orException : public std::exception
{
public:
	orException(void);
	orException(std::string msg);
	virtual ~orException(void) throw(){};

	void setMessage(std::string msg);

public:
	std::string	message;
};


// Exception related to OpenCV Functions
class orCvException : public orException
{
public:
	orCvException(void);
	orCvException(std::string msg);
	virtual ~orCvException(void) throw(){};

	void setFunctionName(std::string name);
	void setCvExceptionClass(cv::Exception e);

public:
	std::string cvfunction;
	cv::Exception cv_e;
};

// Illegal Argument
class orArgException : public orException
{
public:
	orArgException(void);
	orArgException(std::string msg);
	virtual ~orArgException(void) throw(){};

	void setFunctionName(std::string name);

public:
	std::string function;
};


// State Error
class orStateException : public orException
{
public:
	orStateException(void);
	orStateException(std::string msg);
	virtual ~orStateException(void) throw(){};
};

};

#endif
