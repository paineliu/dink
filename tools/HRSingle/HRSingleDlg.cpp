
// HRSingleDlg.cpp : implementation file
//

#include "pch.h"
#include "framework.h"
#include "HRSingle.h"
#include "HRSingleDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define HR_CAND_MAX_NUM 8

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CHRSingleDlg dialog



CHRSingleDlg::CHRSingleDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_HRSINGLE_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CHRSingleDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_STATIC_CAND, m_ctrlCand);
}

BEGIN_MESSAGE_MAP(CHRSingleDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_BN_CLICKED(IDC_BUTTON_LOAD, &CHRSingleDlg::OnBnClickedButtonLoad)
	ON_BN_CLICKED(IDC_BUTTON_UNDO, &CHRSingleDlg::OnBnClickedButtonUndo)
	ON_BN_CLICKED(IDC_BUTTON_SAVE, &CHRSingleDlg::OnBnClickedButtonSave)
	ON_BN_CLICKED(IDC_BUTTON_CLEAN, &CHRSingleDlg::OnBnClickedButtonClean)
	ON_BN_CLICKED(IDC_BUTTON_RECOG, &CHRSingleDlg::OnBnClickedButtonRecog)
	ON_BN_CLICKED(IDCANCEL, &CHRSingleDlg::OnBnClickedCancel)
	ON_BN_CLICKED(IDOK, &CHRSingleDlg::OnBnClickedOk)
END_MESSAGE_MAP()


// CHRSingleDlg message handlers

BOOL CHRSingleDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here
	m_nStrokeNum = 0;
	m_szCand[0] = 0;
	m_hDink = dink_init("casia.onx");
	
	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CHRSingleDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CHRSingleDlg::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	if (IsIconic())
	{
		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
		if (m_vPoints.size() > 0)
		{
			for (int i=0; i<(int) m_vPoints.size(); i++)
			{
				if (i == 0 || m_vPoints[i].s != m_vPoints[i - 1].s)
				{
					dc.MoveTo(m_vPoints[i].x, m_vPoints[i].y);
				}
				else
				{
					dc.LineTo(m_vPoints[i].x, m_vPoints[i].y);
				}
			}
		}
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CHRSingleDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CHRSingleDlg::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default
	m_bPress = TRUE;
	SetCapture();
	m_nStrokeNum++;
	DK_POINT pt;

	pt.x = point.x;
	pt.y = point.y;
	pt.s = m_nStrokeNum;
	pt.t = GetTickCount();

	m_vPoints.push_back(pt);
	CDialogEx::OnLButtonDown(nFlags, point);
}


void CHRSingleDlg::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default
	if (m_bPress)
	{
		DK_POINT pt;

		pt.x = point.x;
		pt.y = point.y;
		pt.s = m_nStrokeNum;
		pt.t = GetTickCount();

		m_vPoints.push_back(pt);

		ReleaseCapture();

		OnRecognize();

		m_bPress = FALSE;
	}
	CDialogEx::OnLButtonUp(nFlags, point);
}


void CHRSingleDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default
	if (m_bPress)
	{
		DK_POINT pt;

		pt.x = point.x;
		pt.y = point.y;
		pt.s = m_nStrokeNum;
		pt.t = GetTickCount();

		m_vPoints.push_back(pt);

		Invalidate(FALSE);
	}
	CDialogEx::OnMouseMove(nFlags, point);
}


void CHRSingleDlg::OnBnClickedButtonLoad()
{
	// TODO: Add your control notification handler code here
	TCHAR szFilters[] = _T("Text Type Files (*.txt)|*.txt|All Files (*.*)|*.*||");

	CFileDialog dlg(TRUE, NULL, _T("*.txt"), OFN_FILEMUSTEXIST | OFN_HIDEREADONLY, szFilters);

	if (dlg.DoModal() == IDOK)
	{
		TCHAR szFileName[MAX_PATH];
		_tcscpy_s(szFileName, dlg.GetPathName().LockBuffer());
		dlg.GetPathName().ReleaseBuffer();
		char szLine[1024];
		FILE* fp = NULL;
		_tfopen_s(&fp, szFileName, _T("r"));
		int x;
		int y;
		int s;
		int t;
		DK_POINT stPoint;
		m_vPoints.clear();
		std::vector<DK_POINT> vPoint;
		if (fp)
		{
			while (fgets(szLine, 1024, fp))
			{
				if (sscanf_s(szLine, "%d, %d, %d, %d", &x, &y, &s, &t) == 4)
				{
					stPoint.x = x;
					stPoint.y = y;
					stPoint.s = s;
					stPoint.t = t;
					m_vPoints.push_back(stPoint);
				}
			}
			fclose(fp);
		}

		OnRecognize();
	}

}


void CHRSingleDlg::OnBnClickedButtonUndo()
{
	// TODO: Add your control notification handler code here
	if (m_nStrokeNum > 0)
	{
		for (size_t i = 0; i < m_vPoints.size(); i++)
		{
			if (m_vPoints[i].s == m_nStrokeNum)
			{
				m_vPoints.resize(i);
				break;
			}
		}

		m_nStrokeNum--;
		
		m_ctrlCand.SetWindowText(_T(""));

		if (m_nStrokeNum > 0)
		{
			OnRecognize();
		}

		Invalidate(TRUE);
	}
}


void CHRSingleDlg::OnBnClickedButtonSave()
{
	// TODO: Add your control notification handler code here
	TCHAR szFilters[] = _T("Text Type Files (*.txt)|*.txt|All Files (*.*)|*.*||");
	TCHAR szFileName[64];
	wsprintf(szFileName, _T("%s.txt"), m_szCand);
	CFileDialog dlg(FALSE, NULL, szFileName, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY, szFilters);

	if (dlg.DoModal() == IDOK)
	{
		TCHAR szFileName[MAX_PATH];
		_tcscpy_s(szFileName, dlg.GetPathName().LockBuffer());
		dlg.GetPathName().ReleaseBuffer();

		FILE* fp = NULL;
		_tfopen_s(&fp, szFileName, _T("w"));
		if (fp)
		{
			int nPointTotal = m_vPoints.size();
			for (int i = 0; i < nPointTotal; i++)
			{
				fprintf(fp, "%d, %d, %d, %d\n", m_vPoints[i].x, m_vPoints[i].y, m_vPoints[i].s, m_vPoints[i].t);
			}

			fclose(fp);
		}
	}
}


void CHRSingleDlg::OnBnClickedButtonClean()
{
	// TODO: Add your control notification handler code here
	m_ctrlCand.SetWindowText(_T(""));
	m_vPoints.clear();
	m_nStrokeNum = 0;
	Invalidate(TRUE);
}


void CHRSingleDlg::OnBnClickedButtonRecog()
{
	OnRecognize();
}


void CHRSingleDlg::OnRecognize()
{
	unsigned short nCand[HR_CAND_MAX_NUM];
	dink_recog(m_hDink, m_vPoints.data(), m_vPoints.size(), nCand, HR_CAND_MAX_NUM);
	for (int i = 0; i < HR_CAND_MAX_NUM; i++)
	{
		m_szCand[i * 2] = nCand[i];
		m_szCand[i * 2 + 1] = ' ';
	}
	m_szCand[2 * HR_CAND_MAX_NUM - 1] = 0;

	Invalidate(FALSE);
	m_ctrlCand.SetWindowText(m_szCand);
}

void CHRSingleDlg::OnBnClickedCancel()
{
	// TODO: Add your control notification handler code here
	CDialogEx::OnCancel();
}


void CHRSingleDlg::OnBnClickedOk()
{
	// TODO: Add your control notification handler code here
	CDialogEx::OnOK();
}
