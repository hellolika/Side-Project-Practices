CREATE TABLE [dbo].[LeaveAmount]
(
	[Id] INT NOT NULL PRIMARY KEY IDENTITY (1,1),
	[MemberId] INT NOT NULL,
	[LeaveType] INT NOT NULL,
	[RemainingLeaves] DECIMAL(16,9) NOT NULL,
	[LeavesGranted] DECIMAL(16,9) NOT NULL,
	[Year] INT NOT NULL,
	[ModifyBy] INT NULL,
	[ModifyOn] DATETIME NULL,
)
