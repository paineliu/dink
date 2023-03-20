
// HRSingleDlg.h : header file
//

#pragma once

#include "dink.h"
#include <vector>

// CHRSingleDlg dialog
class CHRSingleDlg : public CDialogEx
{
// Construction
public:
	CHRSingleDlg(CWnd* pParent = nullptr);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_HRSINGLE_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;
	std::vector<DK_POINT> m_vPoints;
	DK_HANDLE m_hDink;
	int m_nStrokeNum;
	BOOL m_bPress;
	CPoint m_lastPoint;
	WCHAR m_szCand[64];
	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnBnClickedButtonLoad();
	afx_msg void OnBnClickedButtonUndo();
	afx_msg void OnBnClickedButtonSave();
	afx_msg void OnBnClickedButtonClean();
	afx_msg void OnBnClickedButtonRecog();

private:
	void OnRecognize();
public:
	CStatic m_ctrlCand;
	afx_msg void OnBnClickedCancel();
	afx_msg void OnBnClickedOk();
};
