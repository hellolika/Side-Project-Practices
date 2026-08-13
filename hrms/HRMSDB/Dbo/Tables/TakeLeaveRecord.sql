CREATE TABLE [dbo].[TakeLeaveRecord]
(
	[RequestId] INT NOT NULL PRIMARY KEY IDENTITY(1, 1),
	[MemberId] INT,
	[NumberOfDay] DECIMAL(16,9) NULL,
	[StartDate] DATETIME NOT NULL,
	[EndDate] DATETIME NOT NULL,
	[LeaveType] INT NOT NULL,
	[Image] VARCHAR(100) NULL,
	[Reason] NVARCHAR(500) NOT NULL,
	[ResponseReason] NVARCHAR(500) NULL,
	[IsCancel] BIT NOT NULL DEFAULT 0,
	[Status] INT NOT NULL DEFAULT 0,
	[SubmittedOn] DATETIME DEFAULT GETDATE(),
	[UpdateBy] INT,
	[UpdateOn] DATETIME,
)
