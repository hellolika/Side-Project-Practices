CREATE TABLE [dbo].[LeaveType]
(
	[TypeId] INT NOT NULL PRIMARY KEY IDENTITY(0, 1),
	[Type] NVARCHAR(50) NOT NULL,
	[DefaultLeavesGranted] DECIMAL(16,9) DEFAULT 0,
	[IsEnable] BIT DEFAULT 1,
	[IsLimited] BIT DEFAULT 1
)
