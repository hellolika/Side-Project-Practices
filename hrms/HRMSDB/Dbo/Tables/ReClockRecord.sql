CREATE TABLE [dbo].[ReClockRecord]
(
	[RequestId] INT NOT NULL PRIMARY KEY IDENTITY(1, 1),
	[MemberId] INT NOT NULL,
	[Date] DATE NOT NULL,
	[Time] TIME(0) NOT NULL,
	[IsClockIn] BIT NOT NULL DEFAULT 1,
	[Reason] NVARCHAR(200) NULL,
	[ResponseReason] NVARCHAR(200) NULL,
	[Location] NVARCHAR(200) NULL,
	[Status] INT NOT NULL DEFAULT 0,
	[UpdateBy] INT,
)
